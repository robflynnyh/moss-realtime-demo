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

The default runtime profile is `throughput`, which optimizes generated audio
seconds per wall second. For live stdin/text streaming, use the interactive
profile:

```bash
printf 'This text arrives through stdin and is chunked into deltas for realtime synthesis.\n' \
  | scripts/run_streaming_demo.sh \
      --runtime-profile interactive \
      --prompt-wav prompts/fdr_fireside_12s.wav
```

Outputs default to timestamped WAV files in `outputs/`.

## Benchmark

Add `--benchmark` to record speed metrics. It prints a short timing summary and
writes JSON next to the WAV as `<output>.benchmark.json`.

```bash
scripts/run_streaming_demo.sh \
  --benchmark \
  --warmup-runs 1 \
  --runtime-profile throughput \
  --prompt-wav prompts/nabu_joe_en_us_12s.wav \
  --out-wav outputs/bench_joe.wav \
  --text "Benchmarking the realtime streaming path with a short test sentence."
```

The metrics include setup timings, model/codec load time, prompt encoding time,
time to first generated audio tokens, time to first decoded audio chunk, chunk
interval p50/p95/max, output duration, total streaming time, generated audio
seconds per wall-clock second, realtime factor, and a stage breakdown for model
streaming calls versus codec waveform decoding.

The demo also has an experimental async codec path:

```bash
scripts/run_streaming_demo.sh \
  --benchmark \
  --async-codec-decode \
  --prompt-wav prompts/nabu_joe_en_us_12s.wav \
  --out-wav outputs/bench_async_codec.wav \
  --text "Benchmarking asynchronous codec decoding with the same realtime text stream."
```

It queues audio-token batches to a worker thread while the main thread continues
token generation. On this RTX A4500, deterministic short-run tests were slower
with async decode (`0.499` audio seconds per wall second) than the serial
default (`0.519`), because concurrent model and codec kernels contend on the
same GPU. Keep serial decode as the default here, and use `--async-codec-decode`
only for comparison or on hardware with more scheduling headroom.

For more accurate stage attribution, add `--benchmark-synchronize`. It inserts
CUDA synchronizations around profiled model and codec stages, so it can perturb
absolute latency but better answers where time is being spent.

Compare multiple benchmark JSON files with:

```bash
scripts/compare_benchmarks.py outputs/*.benchmark.json
```

`--runtime-profile throughput` is now the default. It feeds full text in one
delta, drains the audio-token model in large batches, disables sampling, and
decodes the waveform at flush time. On the deterministic short A4500 test this
profile reached `0.672` generated audio seconds per wall second versus `0.531`
for the best interactive drain profile.

Use `--runtime-profile interactive` when first-audio/chunk cadence matters. The
interactive profile uses smaller text deltas, `--decode-chunk-frames 3`,
`--drain-max-steps 3`, `--first-drain-max-steps 1`, and sampling enabled.

The runtime path uses BF16 for model weights when CUDA supports it, local-transformer
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
over one second between chunks, and `3` was the best interactive middle ground.
The throughput profile sets this to `4096`, which effectively decodes once at
the end.

`--drain-max-steps` controls how many autoregressive audio-token steps are run
per post-text drain call after the first audio chunk. The default is `3` with
`--first-drain-max-steps 1`: this keeps first-audio latency conservative while
reducing Python/GPU round trips after audio has started. On the deterministic
short A4500 test, drain `3` cut drain calls from `74` to `25`, improved
throughput from `0.519` to `0.531` generated audio seconds per wall second, and
kept chunk p95 near `0.45s`.

## RL Rollout Batching

For offline RL rollout generation, use the batched runner:

```bash
scripts/run_batch_rollout.sh \
  --benchmark \
  --batch-size 2 \
  --max-audio-steps 512 \
  --prompt-wav prompts/nabu_joe_en_us_12s.wav \
  --text "First full rollout text goes here." \
  --text "Second full rollout text goes here."
```

It bypasses the single-stream `MossTTSRealtimeStreamingSession` and calls the
lower-level batch-shaped `MossTTSRealtimeInference.prefill()` and `step()` APIs
directly. The default `--packing-mode interleaved` matches the latency-oriented
streaming scheme: prefill the first text tokens, then repeatedly step a vector
of the next text token per sample plus the previous audio tokens, using text-pad
tokens for samples whose text is exhausted. It then drains audio-token
generation until EOS or `--max-audio-steps`, decodes all samples with one codec
`batch_decode()` call per microbatch, and writes WAVs plus `manifest.json` under
the output directory.

Use `--prefill-text-len` to override the initial text prefix length. By default
it uses the processor delay length, currently `12` tokens. The older
throughput-only behavior is still available as `--packing-mode full-text`; that
prefills the full text before audio-token drain and is not the deployment
streaming layout.

Waveform decode is subbatched by default with `--codec-decode-batch-size 16`.
This keeps large rollout batches from OOMing in codec `batch_decode()` after
audio-token generation has already succeeded. Pass `--codec-decode-batch-size 0`
to decode the full microbatch at once when you know it fits.

You can also pass a file:

```bash
scripts/run_batch_rollout.sh \
  --benchmark \
  --batch-size 2 \
  --texts-file rollout_texts.jsonl
```

Plain text files are read as one rollout per non-empty line. JSONL files should
contain a `text` field and may include an `id` field for the output filename.
The benchmark manifest separates setup, prompt encode, text prefill,
interleaved text/audio stepping, codec batch decode, WAV writing, and aggregate
generated audio seconds per wall second. It also records packing mode,
prefill/stepped text-token counts, drain-step counts, and peak CUDA
allocated/reserved memory per microbatch, including separate peaks for prefill,
interleaved token generation, and codec decode.

On the 20 GB RTX A4500, a longer interleaved sweep with about `166` text tokens
per sample and `--max-audio-steps 192` generated about `15.44s` of audio per
sample. This shape had `163` text-stepping calls and `29` final drain calls.

| batch size | codec decode batch | audio seconds / batch wall second | peak allocated | peak reserved |
| --- | ---: | ---: | ---: | ---: |
| 16 | full | 10.253 | 12.9 GiB | 14.0 GiB |
| 32 | full | 19.458 | 14.7 GiB | 17.4 GiB |
| 64 | full | OOM in codec decode | - | - |
| 64 | 16 | 35.131 | 14.9 GiB | 18.4 GiB |
| 96 | 16 | 47.913 | 16.3 GiB | 19.5 GiB |

For the successful `96` run, the per-stage peaks were `14.1 GiB` prefill,
`15.2 GiB` interleaved token generation, and `16.3 GiB` codec decode. So for
this shape, codec decode was the largest allocation stage, but generation is
also close enough that `96` should be treated as a high-throughput edge setting.
Use `64` for more headroom or longer outputs.

An earlier short fixed-shape sweep with
`--packing-mode full-text --max-audio-steps 64` kept improving up to batch size
`96`. Batch size `128` OOMed during codec waveform `batch_decode()`, not during
audio-token generation. Treat `96` as the measured high-throughput cap for
short full-text rollouts, and use `64` when you want more memory headroom or
longer outputs.

| batch size | audio seconds / batch wall second | peak allocated | peak reserved |
| --- | ---: | ---: | ---: |
| 1 | 0.654 | 11.1 GiB | 11.6 GiB |
| 2 | 1.303 | 11.2 GiB | 11.6 GiB |
| 4 | 2.642 | 11.3 GiB | 11.7 GiB |
| 8 | 5.214 | 11.5 GiB | 12.0 GiB |
| 16 | 9.700 | 11.9 GiB | 12.6 GiB |
| 32 | 18.166 | 12.7 GiB | 13.7 GiB |
| 64 | 33.345 | 14.2 GiB | 16.0 GiB |
| 96 | 43.179 | 15.8 GiB | 18.1 GiB |
| 128 | OOM in codec decode | 16.6 GiB allocated before OOM | 19.3 GiB process use |

For longer rollouts, rerun the sweep with your target `--max-audio-steps`; KV
cache, generated-token history, and codec decode memory all grow with output
length.

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
