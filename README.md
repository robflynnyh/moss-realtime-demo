# MOSS-TTS Realtime Streaming Demo

This directory wraps the upstream MOSS-TTS realtime example in a local stdin/text
demo. It streams text deltas into MOSS-TTS-Realtime and writes the generated WAV
under `./outputs`.

The upstream checkout is in `vendor/MOSS-TTS`; local demo code lives in
`scripts/`.

## Setup

Fetch the upstream code submodules after cloning:

```bash
git submodule update --init --recursive
```

The environment has already been created in `.venv` with Python 3.12 and CUDA
12.1 PyTorch wheels, which match this server's NVIDIA driver better than the
README's CUDA 12.8 wheel example.

To recreate it:

```bash
uv venv --python 3.12 .venv
UV_CACHE_DIR=$PWD/.uv-cache uv pip install --python .venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
  transformers==5.0.0 accelerate soundfile -e vendor/MOSS-TTS
```

## Prompt Samples

Download and trim the public-domain prompt samples:

```bash
scripts/download_prompt_samples.sh
```

The generated WAVs are written to `prompts/`. The set includes archival
public-domain speeches plus cleaner CC0 voice-dataset samples from Nabu Casa.

## Run

Use the cooperative GPU scheduler on this server:

```bash
scripts/run_streaming_demo.sh \
  --prompt-wav prompts/jfk_berlin_12s.wav \
  --text "Welcome to the realtime MOSS streaming demo. This sentence is fed to the model in small text chunks."
```

For stdin streaming:

```bash
printf 'This text arrives through stdin and is chunked into deltas for realtime synthesis.\n' \
  | scripts/run_streaming_demo.sh --prompt-wav prompts/fdr_fireside_12s.wav
```

Outputs default to timestamped WAV files in `outputs/`.

## Benchmark

Add `--benchmark` to record speed metrics. It prints a short timing summary and
writes JSON next to the WAV as `<output>.benchmark.json`.

```bash
scripts/run_streaming_demo.sh \
  --benchmark \
  --warmup-runs 1 \
  --prompt-wav prompts/nabu_joe_en_us_12s.wav \
  --out-wav outputs/bench_joe.wav \
  --text "Benchmarking the realtime streaming path with a short test sentence."
```

The metrics include setup timings, model/codec load time, prompt encoding time,
time to first generated audio tokens, time to first decoded audio chunk, chunk
interval p50/p95/max, output duration, total streaming time, generated audio
seconds per wall-clock second, realtime factor, and a stage breakdown for model
streaming calls versus codec waveform decoding.

For more accurate stage attribution, add `--benchmark-synchronize`. It inserts
CUDA synchronizations around profiled model and codec stages, so it can perturb
absolute latency but better answers where time is being spent.

Compare multiple benchmark JSON files with:

```bash
scripts/compare_benchmarks.py outputs/*.benchmark.json
```

The default runtime path is optimized for the local RTX A4500 measurements: BF16
is used for model weights when CUDA supports it, local-transformer
`torch.compile` is off by default because it was much slower for this streaming
shape pattern, and TF32 is allowed for remaining FP32 CUDA matmuls unless
`--no-allow-tf32` is passed. You can still test compilation with
`--local-compile --warmup-runs 1`; use warmup runs so compilation cost is
excluded from the measured/output run. Compile controls are exposed via
`--local-compile-mode`, `--local-compile-fullgraph`, `--local-compile-dynamic`,
and `--local-compile-backend`. For controlled compile tests, start with
`--repetition-penalty 1.0` to remove the dynamic generated-history penalty path,
then compare quality/speed before restoring the default penalty.

`--decode-chunk-frames` controls latency/throughput tradeoff in the audio
token-to-waveform decoder. On the short A4500 benchmark, `1` gave the fastest
first/chunk cadence but slowest throughput, `8` gave the best throughput but
over one second between chunks, and the default `3` was the best interactive
middle ground.

FlashAttention 2 was tested in a separate `.venv-fa2` environment so the stable
demo `.venv` stayed untouched. It imports and runs, but on the local RTX A4500
benchmarks it was slower than the default SDPA path:

```bash
MOSS_DEMO_VENV=$PWD/.venv-fa2 scripts/run_streaming_demo.sh \
  --benchmark \
  --attn-implementation flash_attention_2 \
  --prompt-wav prompts/nabu_joe_en_us_12s.wav \
  --out-wav outputs/bench_fa2.wav \
  --text "Benchmarking the realtime streaming path with a short test sentence."
```

## SGLang-Omni

SGLang-Omni is the optimized serving path for `MOSS-TTS-Local-Transformer-v1.5`,
but the current stack is not runnable on this server's installed driver. An
isolated `.venv-sglang` install imports successfully, but its CUDA 13 PyTorch
build reports the NVIDIA driver is too old and `torch.cuda.is_available()` is
false. Keep the working realtime demo on `.venv`.

See `docs/sglang.md` for the exact install attempt, dependency pins, driver
check, and next options.
