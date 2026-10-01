"""LoRA DDP or matched language-linear full-rank FSDP training."""
import argparse
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .data import read_rows, resolve_data, save_json
from .evaluation import evaluate_predictions
from .learning import Readout, loss_terms
from .prompts import option_labels, prompt
from .scheduling import batch_plan, split_batch, lr_factor, select_checkpoint


def main():
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--base-model')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    modules_path = root / cfg['modules']
    targets = json.loads(modules_path.read_text())
    train_path, dev_path = resolve_data(cfg['dataset'], args.data_dir)
    rows = read_rows(train_path, training=True)
    dev = read_rows(dev_path, evaluation=True)
    if len(rows) != cfg['training_rows']:
        raise ValueError('The dataset size does not match the release configuration.')
    base_path = args.base_model or cfg['base_model']
    base_kwargs = {} if args.base_model else {'revision': cfg.get('base_revision')}
    tokenizer = AutoTokenizer.from_pretrained(base_path, **base_kwargs)
    tokenizer.padding_side = 'right'
    labels = option_labels(tokenizer)
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]
    lengths = [len(tokenizer.encode(prompt(tokenizer, row, labels), add_special_tokens=False)) for row in rows]
    dev_lengths = [len(tokenizer.encode(prompt(tokenizer, row, labels), add_special_tokens=False)) for row in dev]
    if max(lengths) > cfg['max_input_tokens'] or max(dev_lengths) > cfg['max_input_tokens']:
        raise ValueError('An input exceeds the token limit; inputs are never truncated.')
    batches = batch_plan(rows, lengths, cfg['seed'], cfg['epochs'])
    planned_steps = len(batches)
    total = min(planned_steps, cfg['checkpoint_steps'])
    if args.validate_only:
        print(json.dumps(dict(train_n=len(rows), validation_n=len(dev), planned_steps=planned_steps,
                              checkpoint_steps=total, max_train_tokens=max(lengths),
                              modules=len(targets), option_labels=len(labels))), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA. Use --validate-only for CPU validation.')
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world != cfg['gpus'] or 64 % world:
        raise ValueError(f"Use torchrun with {cfg['gpus']} GPUs for this configuration.")
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local)
    torch.manual_seed(cfg['seed'])
    output = args.output
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'training_complete.json').exists():
            raise ValueError('The output already contains a completed run.')
    dist.barrier()
    conf = AutoConfig.from_pretrained(base_path, **base_kwargs)
    cls = Qwen3_5ForConditionalGeneration if conf.model_type == 'qwen3_5' else Qwen3_5ForCausalLM
    full = cfg['method'] == 'full'
    base = cls.from_pretrained(base_path, dtype=torch.float32 if full else torch.bfloat16,
                              device_map='cpu' if full else str(device), attn_implementation='sdpa',
                              **base_kwargs)
    base.requires_grad_(False)
    if full:
        modules = dict(base.named_modules())
        for name in targets:
            if not isinstance(modules[name], torch.nn.Linear):
                raise TypeError('The full-rank module list must contain language linear layers.')
            modules[name].requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(cfg['seed'])
        base = get_peft_model(base, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.,
                                              target_modules=targets, bias='none'))
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    if trainable != cfg['trainable_parameters']:
        raise ValueError('The trainable parameter count differs from the release configuration.')
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    base.config.use_cache = False
    wrapper = Readout(base, label_ids, full)
    if full:
        model = FSDP(wrapper, auto_wrap_policy=partial(transformer_auto_wrap_policy,
                     transformer_layer_cls={type(base.model.language_model.layers[0])}),
                     mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                     reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16),
                     device_id=device, use_orig_params=True, limit_all_gathers=True)
    else:
        model = DDP(wrapper.to(device), device_ids=[local], broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg['lr'], betas=(.9, .999), eps=1e-8,
                                 weight_decay=0., foreach=False)
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output / 'tensorboard')
        save_json(output / 'training_config.json', dict(**cfg, planned_steps=planned_steps))
    entries = []
    dev_order = sorted(range(len(dev)), key=lambda i: (dev_lengths[i], dev[i].get('validation_order', dev[i]['id'])))
    evaluate_steps = {math.ceil(planned_steps*i/10) for i in range(1, 11)} | {total}

    def encode(part):
        encoded = tokenizer([prompt(tokenizer, row, labels) for row in part], padding=True,
                            return_tensors='pt', add_special_tokens=False)
        return {key: value.to(device) for key, value in encoded.items()}

    @torch.no_grad()
    def evaluate(step):
        model.eval()
        predictions = []
        offset = 0
        while offset < len(dev_order):
            micro = 8
            maximum = max(dev_lengths[i] for i in dev_order[offset:offset+world*micro])
            while micro > 1 and micro*maximum > 8192:
                micro //= 2
            indices = dev_order[offset:offset+world*micro]
            count = len(indices)
            padded = indices + [indices[-1]] * (world*micro-count)
            own = padded[rank*micro:(rank+1)*micro]
            logits = model(**encode([dev[i] for i in own]))
            for j, index in enumerate(own):
                if rank*micro+j < count:
                    predictions.append(dict(id=dev[index]['id'],
                                            probs=logits[j, :len(dev[index]['options'])].softmax(-1).cpu().tolist()))
            offset += count
        gathered = [None] * world
        dist.all_gather_object(gathered, predictions)
        result = evaluate_predictions(dev, [p for part in gathered for p in part])
        result['step'] = step
        if rank == 0:
            save_json(output / f'validation_{step:06d}.json', result)
            writer.add_scalar('validation/score', result['score'], step)
            writer.add_scalar('validation/nll', result['nll'], step)
            writer.flush()
        model.train()
        return dict(step=step, score=result['score'], nll=result['nll'])

    def export(step):
        destination = output / f'checkpoint-{step}'
        state = None
        if full:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                     FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                state = model.state_dict()
        if rank == 0:
            if full:
                exported = {key.removeprefix('base.'): value.to(dtype=torch.bfloat16)
                            for key, value in state.items()}
                base.save_pretrained(destination, state_dict=exported,
                                    safe_serialization=True, max_shard_size='5GB')
            else:
                base.save_pretrained(destination, safe_serialization=True)
            tokenizer.save_pretrained(destination)
        del state
        dist.barrier()

    model.train()
    for index, indices in enumerate(batches[:total]):
        step = index+1
        begin = time.monotonic()
        count = len(indices)
        for group in optimizer.param_groups:
            group['lr'] = cfg['lr'] * lr_factor(step, planned_steps)
        optimizer.zero_grad(set_to_none=True)
        sums = torch.zeros(5, device=device, dtype=torch.float64)
        for part in split_batch(indices, lengths, world, cfg['micro'])[rank]:
            batch = [rows[i] for position, i in part]
            terms = loss_terms(model(**encode(batch)), batch, cfg['type_weights'])
            active = torch.tensor([position < count for position, i in part], device=device)
            (terms[0]*active).sum().mul(world/count).backward()
            for j, term in enumerate(terms):
                sums[j] += (term.detach()*active).sum()
            sums[4] += active.sum()
        dist.all_reduce(sums)
        grad = model.clip_grad_norm_(1.) if full else torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(sums).all() or not torch.isfinite(grad):
            raise FloatingPointError('Non-finite loss or gradient.')
        optimizer.step()
        record = dict(step=step, planned_steps=planned_steps, loss=float(sums[0]/sums[4]),
                      ce=float(sums[1]/sums[4]), score_loss=float(sums[2]/sums[4]),
                      kl=float(sums[3]/sums[4]), grad=float(grad), lr=optimizer.param_groups[0]['lr'],
                      seconds=time.monotonic()-begin)
        if rank == 0:
            with (output / 'training.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            for key in ['loss', 'ce', 'score_loss', 'kl', 'lr']:
                writer.add_scalar('train/'+key, record[key], step)
            print(json.dumps(record), flush=True)
        if step in evaluate_steps:
            optimizer.zero_grad(set_to_none=True)
            entries.append(evaluate(step))
            chosen = select_checkpoint(entries)
            if step == total or chosen['step'] == step:
                export(step)
            if rank == 0:
                save_json(output / 'selection.json', chosen)
    if rank == 0:
        save_json(output / 'training_complete.json', dict(steps=total, selected=select_checkpoint(entries)))
        writer.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
