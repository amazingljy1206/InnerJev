#!/usr/bin/env python3
"""Merge a LoRA adapter into a standalone BF16 Hugging Face model."""
import argparse
from pathlib import Path


def main():
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.glob('*.safetensors')):
        parser.error('The output already contains model weights.')
    torch.set_num_threads(args.threads)
    config = AutoConfig.from_pretrained(args.base)
    cls = Qwen3_5ForConditionalGeneration if config.model_type == 'qwen3_5' else Qwen3_5ForCausalLM
    base = cls.from_pretrained(args.base, dtype=torch.bfloat16, device_map=args.device,
                              attn_implementation='sdpa')
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(args.output, safe_serialization=True, max_shard_size='5GB')
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.output)
    print('Merged model saved to ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
