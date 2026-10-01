"""Read decision probabilities with BF16, NF4, or INT8 weights."""
import argparse
import json
import os
from pathlib import Path

from .data import read_rows
from .prompts import option_labels, prompt


class DecisionEngine:
    def __init__(self, model_path, precision='bf16', gpus=1, gpu_index=0,
                 gpu_memory='20GiB', cpu=False):
        import torch
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
        from transformers import Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
        self.torch = torch
        self.name = model_path.rstrip('/').split('/')[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = 'right'
        self.labels = option_labels(self.tokenizer)
        if cpu:
            if precision != 'bf16' or gpus != 1:
                raise ValueError('CPU smoke inference supports BF16 only.')
            mapping, memory = 'cpu', None
        else:
            if not torch.cuda.is_available() or torch.cuda.device_count() < max(gpus, gpu_index+1):
                raise RuntimeError('The requested CUDA devices are not available.')
            torch.cuda.set_device(gpu_index)
            mapping = {'': gpu_index} if gpus == 1 else 'auto'
            memory = None if gpus == 1 else {i: gpu_memory for i in range(gpus)}
        config = AutoConfig.from_pretrained(model_path)
        cls = Qwen3_5ForConditionalGeneration if config.model_type == 'qwen3_5' else Qwen3_5ForCausalLM
        kwargs = dict(dtype=torch.bfloat16, device_map=mapping, attn_implementation='sdpa')
        if memory is not None:
            kwargs['max_memory'] = memory
        if precision == 'nf4':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        elif precision == 'int8':
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
        elif precision != 'bf16':
            raise ValueError('Unsupported precision.')
        self.model, info = cls.from_pretrained(model_path, output_loading_info=True, **kwargs)
        if info.get('missing_keys') or info.get('mismatched_keys'):
            raise RuntimeError('Incomplete model loading: ' + str(info))
        if not cpu and any(v in ('cpu', 'disk') for v in getattr(self.model, 'hf_device_map', {}).values()):
            raise RuntimeError('The selected precision requires more GPU memory; CPU offload is not enabled.')
        self.model.eval()
        self.backbone = getattr(self.model.model, 'language_model', self.model.model)
        self.head = self.model.get_output_embeddings()
        self.input_device = self.model.get_input_embeddings().weight.device
        self.label_ids = torch.tensor([self.tokenizer.encode(s, add_special_tokens=False)[0] for s in self.labels],
                                      device=self.head.weight.device)
        self.precision = precision

    def predict(self, rows, max_input_tokens=6000):
        torch = self.torch
        encoded = self.tokenizer([prompt(self.tokenizer, row, self.labels) for row in rows],
                                 padding=True, return_tensors='pt', add_special_tokens=False)
        lengths = encoded['attention_mask'].sum(-1)
        if int(lengths.max()) > max_input_tokens:
            raise ValueError('The input exceeds the token limit; no truncation is performed.')
        encoded = {key: value.to(self.input_device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self.backbone(**encoded, use_cache=False, return_dict=True).last_hidden_state
            end = (lengths-1).to(hidden.device)
            slot = hidden[torch.arange(len(rows), device=hidden.device), end].to(self.head.weight.device)
            logits = self.head(slot).float().index_select(-1, self.label_ids)
        output = []
        for i, row in enumerate(rows):
            n = len(row['options'])
            probs = logits[i, :n].softmax(-1).cpu().tolist()
            index = max(range(n), key=probs.__getitem__)
            keys = [o['key'] for o in row['options']]
            result = dict(id=row['id'], model=self.name, type=row['type'], precision=self.precision,
                          probs=probs, probabilities=dict(zip(keys, probs)), predicted_index=index,
                          predicted_key=keys[index], input_tokens=int(lengths[i]))
            if row['type'] == 'noul':
                yes = next((j for j, key in enumerate(keys) if key in ('yes', 'true')), None)
                if yes is not None:
                    result['noul'] = probs[yes]
            if row['type'] == 'score':
                result['score'] = sum(j*p for j, p in enumerate(probs))
                if 'score_values' in row:
                    if len(row['score_values']) != n:
                        raise ValueError('score_values must match the ordered options.')
                    result['expected_score_value'] = sum(v*p for v, p in zip(row['score_values'], probs))
            output.append(result)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--precision', choices=['bf16', 'nf4', 'int8'], default='bf16')
    parser.add_argument('--gpus', type=int, choices=[1, 2], default=1)
    parser.add_argument('--replicas', type=int, choices=[1, 2], default=1)
    parser.add_argument('--gpu-memory', default='20GiB')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-input-tokens', type=int, default=6000)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_input_tokens < 1:
        parser.error('Batch size and token limit must be positive.')
    rows = read_rows(args.input)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    labels = option_labels(tokenizer)
    lengths = [len(tokenizer.encode(prompt(tokenizer, row, labels), add_special_tokens=False)) for row in rows]
    if max(lengths) > args.max_input_tokens:
        raise ValueError('The input exceeds the token limit; no truncation is performed.')
    if args.validate_only:
        print(json.dumps(dict(rows=len(rows), max_input_tokens=max(lengths), option_labels=len(labels))))
        return
    if args.output is None:
        parser.error('--output is required for inference.')
    rank = int(os.environ.get('LOCAL_RANK', '0')) if args.replicas > 1 else 0
    if args.replicas > 1:
        if args.gpus != 1 or int(os.environ.get('WORLD_SIZE', '1')) != args.replicas:
            parser.error('Replicas require torchrun and one GPU per process.')
        rows = rows[rank::args.replicas]
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / f'part-{rank:05d}.jsonl'
    else:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Choose a new output file.')
    engine = DecisionEngine(args.model, args.precision, args.gpus, rank, args.gpu_memory, args.cpu)
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('w') as stream:
        for offset in range(0, len(rows), args.batch_size):
            for prediction in engine.predict(rows[offset:offset+args.batch_size], args.max_input_tokens):
                stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    temporary.replace(output)


if __name__ == '__main__':
    main()
