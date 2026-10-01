# Release validation

All four standalone model folders were loaded with the supplied decision engine on CPU.
Each model completed three synthetic questions covering Choice, Noul and Score; the
returned probability vectors were finite, correctly sized, and normalized.

The two datasets passed JSONL and Arrow validation. Their rendered prompts, teacher
targets, baseline distributions, and deterministic training batch order match the
inputs used to train the released checkpoints. The loss implementation and the
capability-equal validation scoring were compared on CPU with zero numerical difference.

Safetensors shard keys were checked against each model's index. Full-rank tensor files
were copied without changing their values. LoRA-derived models contain merged weights
and do not require adapter loading. File sizes and SHA-256 hashes are provided with
the Hugging Face artifacts and verified after upload.

The single-/dual-RTX-4090 launchers are provided, but RTX 4090 hardware execution has
not been validated in this environment. CPU checks do not measure GPU memory peaks,
quantized benchmark scores, latency or throughput. NF4/INT8 deployment therefore must
not inherit the BF16 benchmark score as a measured quantized result.
