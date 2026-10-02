# InnerJev

**InnerJev-4B** and **InnerJev-27B** are decision models that score ordered user-supplied options.
This repository contains the training implementation, exact adaptation-layer lists,
method configs, LoRA merge utility, and single-/dual-RTX-4090 inference scripts.
Weights and training data are hosted separately on Hugging Face.

## Models

| Model | Method | Training questions | BF16 checkpoint score |
|---|---|---:|---:|
| [InnerJev-4B-LoRA](https://huggingface.co/jylin001206/InnerJev-4B-LoRA) | LoRA | 29,326 | 73.9648 |
| [InnerJev-4B-Full](https://huggingface.co/jylin001206/InnerJev-4B-Full) | Full | 29,326 | 73.4694 |
| [InnerJev-27B-LoRA](https://huggingface.co/jylin001206/InnerJev-27B-LoRA) | LoRA | 39,279 | 80.1517 |
| [InnerJev-27B-Full](https://huggingface.co/jylin001206/InnerJev-27B-Full) | Full | 39,279 | 80.1734 |

Scores use equal weight across the fixed 58-benchmark decision suite and refer to individual
BF16 checkpoints. LoRA scores were measured with the adapter before BF16 merging;
the released LoRA models contain merged weights. Quantized scores are not implied by this table.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[quantization,data]'
```

Python 3.10+ and a CUDA-compatible PyTorch installation are required for GPU execution.
The released model and training-data repositories are public and can be downloaded
without signing in to Hugging Face.

## RTX 4090 inference

Each wrapper accepts `MODEL INPUT.jsonl OUTPUT.jsonl`, followed by optional inference flags.
MODEL can be a Hugging Face repository ID or a local weight folder. Both LoRA-derived and
full-rank models use the same scripts because all released weights are standalone models.

```bash
bash scripts/infer_4b_single_4090.sh jylin001206/InnerJev-4B-LoRA examples/decisions.jsonl out-4b.jsonl
bash scripts/infer_4b_dual_4090.sh jylin001206/InnerJev-4B-Full examples/decisions.jsonl out-4b-dual.jsonl
bash scripts/infer_27b_single_4090.sh jylin001206/InnerJev-27B-LoRA examples/decisions.jsonl out-27b.jsonl
bash scripts/infer_27b_dual_4090.sh jylin001206/InnerJev-27B-Full examples/decisions.jsonl out-27b-dual.jsonl
```

| Model / GPUs | Weight format | Parallelism |
|---|---|---|
| 4B / one 24-GB RTX 4090 | BF16 | One model |
| 4B / two 24-GB RTX 4090 | BF16 | Independent replicas; outputs merged in input order |
| 27B / one 24-GB RTX 4090 | NF4 with double quantization | One quantized model |
| 27B / two 24-GB RTX 4090 | INT8, optionally `--precision nf4` | Model layers balanced across GPUs |

27B BF16 weights exceed the aggregate memory of two 24-GB cards. NF4/INT8 change the
numerical inference path, so the BF16 benchmark scores must not be reported as quantized results.
Set `CUDA_VISIBLE_DEVICES` to select GPUs. No CPU/disk offload is silently enabled.
Batch size defaults to four for 4B and one for 27B. Reduce it for long inputs.

### Input and output

Each JSONL question contains `id`, `type`, `state`, `question`, and ordered
`options: [{"key": "...", "desc": "..."}]`. Types are `choice`, `noul`, or `score`.
Labels are assigned in option order using the original 255 single-token labels;
options are never reordered and no tokenizer tokens are added. String states retain
their formatting; structured states are serialized as sorted-key JSON.

Inference uses the original chat template, thinking disabled, ending in
`The correct answer is (`. A single forward pass supplies the legal option-token
logits, normalized at temperature one. Output includes ordered probabilities,
the selected key, Noul yes probability, or Score expected level as appropriate.
Optional `score_values` maps ordered levels to numerical values. Gold labels are unnecessary.
The default input limit is 6,000 tokens; longer inputs are rejected without truncation.

CPU prompt validation does not load weights:

```bash
python -m innerjev.infer --model jylin001206/InnerJev-4B-LoRA   --input examples/decisions.jsonl --validate-only
```

## Training

The two datasets provide same-size teacher distributions `train_target` and original
non-thinking distributions `p0`, plus a separate 7,506-question validation split.
Both methods within a size use identical questions, targets, baseline probabilities,
prompt rendering, length-bucket batch order, and loss.

The 4B training split contains 29,326 questions; the 27B split contains 39,279.


The teacher is the original model of the same size, performing two reasoning samples
whose answer-slot distributions are averaged. The student is initialized from the original
model and learns without thinking. Gold targets are used for validation and hard-Noul
identification, not as replacement teacher targets.

The loss is `w·CE(q,p) + 0.5w·I(Score)·((E[p]−E[q])/(K−1))² + κ·KL(p0||p)`.
Choice/Noul/Score weights are 1 / 1.1666667 / 0.8. κ=0.2 for hard binary Noul and 0.05 otherwise.
AdamW uses β=(0.9,0.999), ε=1e-8, weight decay 0, global batch 64, 3% warmup,
then cosine decay and gradient clipping at 1. Inputs are not truncated.

LoRA uses r64/alpha128/dropout0 and learning rate 5e-5 under DDP. Full-rank training
uses learning rate 2e-6 under FSDP, with FP32 trainable/master weights and Adam state,
BF16 forward computation, and FP32 reductions. Full-rank adapts the same language linear
layers as LoRA; embeddings, output head, normalization, and vision layers stay frozen.
The exact 176-layer (4B) and 352-layer (27B) lists are in `configs/`.

The documented seeds and `ordering_key` preserve the one-epoch length-bucket schedule.
Cosine decay is defined over the entire planned epoch; each config stops at the
released validation-selected checkpoint step. Checkpoints are evaluated every 10% of
the full schedule and at the stopping point. Validation averages metric components
within a source, sources within a capability, then capabilities equally. Scores within
0.1 points of the best are tied by lower validation NLL, then earlier step.

```bash
bash scripts/train_4b_lora.sh --output outputs/4B-LoRA
bash scripts/train_4b_full.sh --output outputs/4B-Full
bash scripts/train_27b_lora.sh --output outputs/27B-LoRA
bash scripts/train_27b_full.sh --output outputs/27B-Full
```

The wrappers use 4 / 4 / 4 / 8 training GPUs respectively, matching the provided configs.
These are training configurations for GPUs with enough memory; the 4090 scripts above are
for inference. `--base-model` can specify a local copy of the original model and `--data-dir`
a local folder containing `train.jsonl` and `validation.jsonl`. LoRA training saves adapters;
full-rank training saves standalone BF16 models. Distributed numerical behavior can vary
with GPU kernels and the installed software environment.

### Merge a trained adapter

```bash
python scripts/merge_lora.py --base Qwen/Qwen3.5-4B   --adapter outputs/4B-LoRA/checkpoint-276 --output outputs/InnerJev-4B-LoRA
```

The merge supports CPU or CUDA and exports complete BF16 safetensors.

Release validation: [model/data integrity and inference checks](docs/VALIDATION.md).

## Data repositories

- https://huggingface.co/datasets/jylin001206/InnerJev-4B-Training-Data
- https://huggingface.co/datasets/jylin001206/InnerJev-27B-Training-Data

Model/code license: Apache-2.0. Dataset source materials retain their upstream terms;
see the dataset cards and inventories.

## Hosted API access

We provide hosted **InnerJev-27B-Full** and **InnerJev-4B-Full** APIs powered by
vLLM in BF16 (without quantization), so you can try both models without deploying them locally.

To request a personal API key, complete the
**[API access request form / API 访问申请](https://sii-czxy.feishu.cn/share/base/form/shrcnq2Pf9zBkUe1jFiHBjgUaxe)**
or scan the QR code below. After manual review and approval, we will email your
API key, endpoint URLs, and usage instructions.

<p>
  <a href="https://sii-czxy.feishu.cn/share/base/form/shrcnq2Pf9zBkUe1jFiHBjgUaxe">
    <img src="docs/assets/api-access-qr.png" alt="Scan to apply for InnerJev API access / 扫码申请 API key" width="280">
  </a>
</p>

Each key defaults to **60 inference requests per rolling minute**, **1,000 per UTC day**,
and a **30-day validity period**. Limits are shared across both models; you may reapply
after expiry. Requests require `Authorization: Bearer <API_KEY>`.
The current trial endpoints use HTTP: do not submit sensitive data or share your key.
