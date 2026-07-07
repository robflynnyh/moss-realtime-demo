# SGLang-Omni Viability Notes

Date: 2026-07-07

## Summary

SGLang-Omni is the right optimized serving path for `MOSS-TTS-Local-Transformer-v1.5`, but the current stack does not run on this server's installed NVIDIA driver.

The current working demo should stay on `.venv` and `scripts/run_streaming_demo.sh`. Do not replace `.venv` with `.venv-sglang`.

## What Was Tested

- Current SGLang-Omni checkout: `vendor/sglang-omni` at `b09dc9956f7bd1924b0b6210805eb3423e840287`.
- Isolated environment: `.venv-sglang`.
- Install command used an explicit protobuf resolver override matching the repo's `tool.uv.override-dependencies`:

```bash
UV_CACHE_DIR=.uv-cache uv pip install \
  --python .venv-sglang/bin/python \
  --override <(printf 'protobuf>=6.31.1,<7.0.0\n') \
  -e vendor/sglang-omni
```

Without that override, `uv pip install` cannot resolve `s3prl` and `descript-audiotools` together.

## Result

Imports succeed:

```text
torch 2.11.0+cu130
transformers 5.6.0
sglang 0.5.12.post1
sglang_omni import ok
```

CUDA does not initialize on this host:

```text
torch 2.11.0+cu130 cuda build 13.0
available False
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12020).
```

Host driver from `nvidia-smi`:

```text
NVIDIA-SMI 535.161.07
Driver Version: 535.161.07
CUDA Version: 12.2
```

## Why This Matters

Current SGLang-Omni `pyproject.toml` pins a CUDA 13 stack:

- `torch==2.11.0`
- `sglang==0.5.12.post1`
- `flash-attn-4>=4.0.0b9,<4.0.0b16`
- `nixl-cu13>=1.1.0`
- `mooncake-transfer-engine-cuda13>=0.3.10`
- `torchaudio==2.11.0`
- `torchcodec==0.11.1`

The official `lmsysorg/sglang-omni:dev` image is also CUDA 13.0. Docker daemon access is not available to this user on this server anyway.

An older MOSS Local PR ref, `origin/pr-728`, used `torch==2.9.1` / `sglang==0.5.8`, but that resolves to a CUDA 12.6 torch wheel here and predates much of the newer MOSS Local CUDA-graph/state-pool optimization work. It is not a strong benchmark target for the current optimized path.

## Model Size

Hugging Face file metadata:

- `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`: about 9.12 GB on disk.
- `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`: about 8.50 GB on disk.
- `OpenMOSS-Team/MOSS-TTS-v1.5`: about 17.0 GB on disk.

Even with a compatible driver, single-GPU colocation on a 20 GB RTX A4500 may be tight. The current SGLang-Omni config budgets 90% of a colocated GPU and reserves 15% for codec memory; the documented benchmark numbers are from 2x H100 at concurrency 16.

## Other Serving Backends

OpenMOSS also points to vLLM-Omni for `MossTTSRealtime`, which is closer to the model used by this demo. Dry-run dependency checks do not currently make it a usable accelerated path on this host:

- `vllm-omni==0.24.0` cannot resolve against the CUDA 12.1 torch backend because its dependencies require `torch>=2.7.1`.
- `vllm-omni==0.20.0` resolves, but only by moving to `torch==2.12.1+cu126`, which is still outside this host's driver-compatible CUDA stack.
- `vllm-omni==0.16.0` also requires `torch>=2.7.1` through `cache-dit`.

This leaves the current direct MOSS realtime path as the best option on this server unless the NVIDIA driver/runtime is upgraded.

## Next Practical Options

1. Run SGLang-Omni on a machine/container runtime with a driver compatible with the CUDA 13 PyTorch stack, then benchmark the OpenAI-compatible streaming API.
2. If this server's driver can be upgraded, retry `.venv-sglang` before downloading the Local-Transformer weights.
3. Keep optimizing the current realtime model path here; its synchronized profile shows the autoregressive model streaming calls dominate runtime, with codec decode secondary.
