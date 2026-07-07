#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import wave
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Iterator

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("UV_CACHE_DIR", str(ROOT_DIR / ".uv-cache"))
os.environ.setdefault("PIP_CACHE_DIR", str(ROOT_DIR / ".uv-cache" / "pip"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / ".uv-cache" / "xdg"))
os.environ.setdefault("HF_HOME", str(ROOT_DIR / ".hf-cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(ROOT_DIR / ".hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT_DIR / ".hf-cache" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(ROOT_DIR / ".uv-cache" / "torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT_DIR / ".uv-cache" / "triton"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(ROOT_DIR / ".uv-cache" / "vllm"))
os.environ.setdefault("VLLM_CONFIG_ROOT", str(ROOT_DIR / ".uv-cache" / "vllm-config"))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(ROOT_DIR / ".uv-cache" / "flashinfer-workspace"))

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

REALTIME_DIR = ROOT_DIR / "vendor" / "MOSS-TTS" / "moss_tts_realtime"
if str(REALTIME_DIR) not in sys.path:
    sys.path.insert(0, str(REALTIME_DIR))

transformers_logging.disable_progress_bar()

from example_llm_stream_to_tts import (  # noqa: E402
    SAMPLE_RATE,
    _extract_codes,
    _load_audio,
    decode_audio_frames,
    flush_decoder,
    write_wav,
)
from mossttsrealtime.modeling_mossttsrealtime import MossTTSRealtime  # noqa: E402
from mossttsrealtime.processing_mossttsrealtime import MossTTSRealtimeProcessor  # noqa: E402
from mossttsrealtime.streaming_mossttsrealtime import (  # noqa: E402
    AudioStreamDecoder,
    MossTTSRealtimeInference,
    MossTTSRealtimeStreamingSession,
)


class TimedCodecProxy:
    def __init__(self, codec, metrics: dict[str, object] | None):
        self._codec = codec
        self._metrics = metrics

    def __getattr__(self, name: str):
        return getattr(self._codec, name)

    def decode(self, *args, **kwargs):
        sync_cuda_if_enabled(self._metrics)
        started_at = time.perf_counter()
        try:
            return self._codec.decode(*args, **kwargs)
        finally:
            sync_cuda_if_enabled(self._metrics)
            if self._metrics is not None:
                elapsed = time.perf_counter() - started_at
                self._metrics["codec_decode_s"] = self._metrics.get("codec_decode_s", 0.0) + elapsed
                self._metrics["codec_decode_calls"] = self._metrics.get("codec_decode_calls", 0) + 1


def timed_codec(codec, metrics: dict[str, object] | None):
    if metrics is None:
        return codec
    return TimedCodecProxy(codec, metrics)


def sync_cuda_if_enabled(metrics: dict[str, object] | None) -> None:
    if metrics is not None and metrics.get("benchmark_synchronize") and torch.cuda.is_available():
        torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream text deltas into MOSS-TTS-Realtime and save a WAV."
    )
    parser.add_argument(
        "--runtime-profile",
        choices=["throughput", "interactive", "custom"],
        default="throughput",
        help="Default knob profile. Throughput prioritizes generated audio seconds per wall second; interactive prioritizes first/chunk cadence.",
    )
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--prompt-wav", default=str(ROOT_DIR / "prompts" / "jfk_berlin_12s.wav"))
    parser.add_argument("--out-wav", default=None, help="Defaults to outputs/moss_stream_<timestamp>.wav")
    parser.add_argument("--text", default=None, help="Text to stream. If omitted, text is read from stdin.")
    parser.add_argument("--delta-chunk-chars", type=int, default=None)
    parser.add_argument("--delta-delay-s", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--codec-chunk-duration", type=float, default=0.24)
    parser.add_argument("--decode-chunk-frames", type=int, default=None)
    parser.add_argument("--decode-overlap-frames", type=int, default=None)
    parser.add_argument(
        "--drain-max-steps",
        type=int,
        default=None,
        help="Audio-token generation steps per post-text drain call after the first audio chunk has been emitted.",
    )
    parser.add_argument(
        "--first-drain-max-steps",
        type=int,
        default=None,
        help="Drain steps before the first decoded audio chunk. Keep this low to protect first-audio latency.",
    )
    parser.add_argument(
        "--async-codec-decode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode audio-token batches on a worker thread so token generation can continue while codec chunks are produced.",
    )
    parser.add_argument(
        "--async-codec-queue-size",
        type=int,
        default=2,
        help="Maximum queued audio-token batches for async codec decode. Lower values bound latency; higher values allow more buffering.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    parser.add_argument("--no-sample", dest="no_sample", action="store_true", default=None)
    parser.add_argument("--sample", dest="no_sample", action="store_false")
    parser.add_argument("--max-length", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print and save timing metrics for load, setup, streaming, TTFB, and RTF.",
    )
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Benchmark JSON path. Defaults to <out_wav>.benchmark.json when --benchmark is set.",
    )
    parser.add_argument(
        "--benchmark-synchronize",
        action="store_true",
        help="Synchronize CUDA around profiled stages for more accurate attribution. This can slow the run.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Run this many in-process generations before the measured/output run.",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/CUDNN kernels on Ampere+ GPUs.",
    )
    parser.add_argument(
        "--local-compile",
        action="store_true",
        help="Enable torch.compile on the local transformer. It is off by default because it was slower for this streaming workload on the local A4500.",
    )
    parser.add_argument("--local-compile-backend", default="inductor")
    parser.add_argument(
        "--local-compile-mode",
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument(
        "--local-compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--local-compile-dynamic",
        choices=["auto", "true", "false"],
        default="false",
    )
    parser.add_argument("--dynamo-cache-size-limit", type=int, default=64)
    args = parser.parse_args()
    apply_runtime_profile(args)
    return args


def apply_runtime_profile(args: argparse.Namespace) -> None:
    profiles = {
        "interactive": {
            "delta_chunk_chars": 8,
            "decode_chunk_frames": 3,
            "decode_overlap_frames": 0,
            "drain_max_steps": 3,
            "first_drain_max_steps": 1,
            "async_codec_decode": False,
            "no_sample": False,
        },
        "throughput": {
            "delta_chunk_chars": 10000,
            "decode_chunk_frames": 4096,
            "decode_overlap_frames": 0,
            "drain_max_steps": 512,
            "first_drain_max_steps": 512,
            "async_codec_decode": False,
            "no_sample": True,
        },
        "custom": {
            "delta_chunk_chars": 8,
            "decode_chunk_frames": 3,
            "decode_overlap_frames": 0,
            "drain_max_steps": 3,
            "first_drain_max_steps": 1,
            "async_codec_decode": False,
            "no_sample": False,
        },
    }
    for name, value in profiles[args.runtime_profile].items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def chunk_text(text: str, chunk_chars: int, delay_s: float) -> Iterator[str]:
    step = max(1, chunk_chars)
    for idx in range(0, len(text), step):
        if delay_s > 0 and idx > 0:
            time.sleep(delay_s)
        yield text[idx : idx + step]


def stdin_chunks(chunk_chars: int, delay_s: float) -> Iterator[str]:
    step = max(1, chunk_chars)
    while True:
        chunk = sys.stdin.read(step)
        if chunk == "":
            break
        if delay_s > 0:
            time.sleep(delay_s)
        yield chunk


def make_output_path(out_wav: str | None) -> Path:
    if out_wav:
        path = Path(out_wav).expanduser()
        return path if path.is_absolute() else ROOT_DIR / path
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "outputs" / f"moss_stream_{stamp}.wav"


def load_codec(codec_path: str, device: torch.device):
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True).eval()
    return codec.to(device)


def configure_torch_runtime(args: argparse.Namespace) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    try:
        torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")
    except Exception:
        pass
    try:
        import torch._dynamo as dynamo

        dynamo.config.cache_size_limit = int(args.dynamo_cache_size_limit)
    except Exception:
        pass


def record_elapsed(metrics: dict[str, float], key: str, started_at: float) -> None:
    if metrics is not None:
        metrics[key] = time.perf_counter() - started_at


def add_metric_time(metrics: dict[str, object] | None, key: str, elapsed_s: float) -> None:
    if metrics is not None:
        metrics[key] = metrics.get(key, 0.0) + elapsed_s


def add_metric_count(metrics: dict[str, object] | None, key: str, count: int = 1) -> None:
    if metrics is not None:
        metrics[key] = metrics.get(key, 0) + count


def create_decoder(
    args: argparse.Namespace,
    codec,
    device: torch.device,
    metrics: dict[str, float] | None = None,
) -> AudioStreamDecoder:
    started_at = time.perf_counter()
    decoder = AudioStreamDecoder(
        timed_codec(codec, metrics),
        chunk_frames=args.decode_chunk_frames,
        overlap_frames=args.decode_overlap_frames,
        decode_kwargs={"chunk_duration": -1},
        device=device,
    )
    record_elapsed(metrics, "decoder_init_s", started_at)
    return decoder


def compile_dynamic_arg(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def configure_local_compile(
    inferencer: MossTTSRealtimeInference,
    args: argparse.Namespace,
    metrics: dict[str, object] | None = None,
) -> None:
    if not args.local_compile:
        inferencer._should_compile_local_transformer = False
        return

    inferencer._should_compile_local_transformer = True
    inferencer._compiled_local_transformer = None
    compile_mode = None if args.local_compile_mode == "default" else args.local_compile_mode
    compile_dynamic = compile_dynamic_arg(args.local_compile_dynamic)
    original_impl = inferencer._generate_local_transformer_impl

    def get_runner():
        if inferencer._compiled_local_transformer is None:
            started_at = time.perf_counter()
            inferencer._compiled_local_transformer = torch.compile(
                original_impl,
                backend=args.local_compile_backend,
                mode=compile_mode,
                fullgraph=args.local_compile_fullgraph,
                dynamic=compile_dynamic,
            )
            record_elapsed(metrics, "local_compile_wrapper_create_s", started_at)
        return inferencer._compiled_local_transformer

    inferencer._get_local_transformer_runner = get_runner


def build_session(
    args: argparse.Namespace,
    device: torch.device,
    metrics: dict[str, float] | None = None,
):
    setup_started_at = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if metrics is not None:
        metrics["dtype"] = str(dtype).replace("torch.", "")

    started_at = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    record_elapsed(metrics, "tokenizer_load_s", started_at)

    started_at = time.perf_counter()
    processor = MossTTSRealtimeProcessor(tokenizer)
    record_elapsed(metrics, "processor_init_s", started_at)

    started_at = time.perf_counter()
    model = MossTTSRealtime.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    record_elapsed(metrics, "model_load_s", started_at)

    started_at = time.perf_counter()
    codec = load_codec(args.codec_path, device)
    record_elapsed(metrics, "codec_load_s", started_at)

    with torch.inference_mode():
        started_at = time.perf_counter()
        prompt_audio = _load_audio(Path(args.prompt_wav), target_sample_rate=args.sample_rate)
        record_elapsed(metrics, "prompt_audio_load_s", started_at)
        if metrics is not None:
            metrics["prompt_audio_duration_s"] = float(prompt_audio.shape[-1]) / float(args.sample_rate)

        started_at = time.perf_counter()
        prompt_result = codec.encode(
            prompt_audio.unsqueeze(0).to(device),
            chunk_duration=args.codec_chunk_duration,
        )
        prompt_tokens = _extract_codes(prompt_result).cpu().numpy().squeeze(1)
        record_elapsed(metrics, "prompt_encode_s", started_at)
        if metrics is not None:
            metrics["prompt_token_shape"] = list(prompt_tokens.shape)

    started_at = time.perf_counter()
    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    configure_local_compile(inferencer, args, metrics=metrics)
    inferencer.reset_generation_state(keep_cache=False)

    session = MossTTSRealtimeStreamingSession(
        inferencer,
        processor,
        codec=codec,
        codec_sample_rate=args.sample_rate,
        codec_encode_kwargs={"chunk_duration": args.codec_chunk_duration},
        prefill_text_len=processor.delay_tokens_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window if args.repetition_window > 0 else None,
    )
    session.set_voice_prompt_tokens(prompt_tokens)

    system_prompt = processor.make_ensemble(prompt_tokens)
    assistant_prefix_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    assistant_prefix = np.full(
        (len(assistant_prefix_ids), system_prompt.shape[1]),
        fill_value=processor.audio_channel_pad,
        dtype=np.int64,
    )
    assistant_prefix[:, 0] = assistant_prefix_ids
    input_ids = np.concatenate([system_prompt, assistant_prefix], axis=0)
    session.reset_turn(input_ids=input_ids, include_system_prompt=False, reset_cache=True)
    record_elapsed(metrics, "session_init_s", started_at)

    decoder = create_decoder(args, codec, device, metrics=metrics)
    record_elapsed(metrics, "setup_total_s", setup_started_at)
    return session, codec, decoder, input_ids


def note_audio_chunks(
    chunks: Iterator[np.ndarray],
    metrics: dict[str, float] | None,
) -> Iterator[np.ndarray]:
    for chunk in chunks:
        if metrics is not None:
            now = time.perf_counter()
            if "first_audio_chunk_s" not in metrics:
                metrics["first_audio_chunk_s"] = now - metrics["stream_start_perf"]
            last_chunk_perf = metrics.get("_last_audio_chunk_perf")
            if last_chunk_perf is not None:
                metrics.setdefault("_audio_chunk_intervals_s", []).append(now - last_chunk_perf)
            metrics["_last_audio_chunk_perf"] = now
            metrics["audio_chunk_count"] = metrics.get("audio_chunk_count", 0) + 1
            metrics["largest_audio_chunk_samples"] = max(
                metrics.get("largest_audio_chunk_samples", 0),
                int(chunk.reshape(-1).shape[0]),
            )
        yield chunk


def current_drain_max_steps(args: argparse.Namespace, metrics: dict[str, float] | None) -> int:
    if metrics is not None and "first_audio_chunk_s" not in metrics:
        return max(1, int(args.first_drain_max_steps))
    return max(1, int(args.drain_max_steps))


class AsyncCodecDecoder:
    def __init__(
        self,
        decoder: AudioStreamDecoder,
        codebook_size: int,
        audio_eos_token: int,
        *,
        queue_size: int,
        metrics: dict[str, float] | None = None,
    ):
        self._decoder = decoder
        self._codebook_size = codebook_size
        self._audio_eos_token = audio_eos_token
        self._metrics = metrics
        self._input_queue: Queue[list[torch.Tensor] | None] = Queue(maxsize=max(1, queue_size))
        self._output_queue: Queue[tuple[str, object]] = Queue()
        self._thread = Thread(target=self._worker, name="moss-codec-decoder", daemon=True)
        self._closed = False
        self._thread.start()

    def submit(self, audio_frames: list[torch.Tensor]) -> None:
        if self._closed:
            raise RuntimeError("Async codec decoder is already closed.")
        if self._metrics is not None:
            self._metrics["async_decode_submitted_batches"] = (
                self._metrics.get("async_decode_submitted_batches", 0) + 1
            )
            self._metrics["async_decode_submitted_frames"] = (
                self._metrics.get("async_decode_submitted_frames", 0) + len(audio_frames)
            )
        started_at = time.perf_counter()
        self._input_queue.put(audio_frames)
        add_metric_time(self._metrics, "async_decode_input_queue_wait_s", time.perf_counter() - started_at)

    def available_chunks(self) -> Iterator[np.ndarray]:
        while True:
            try:
                kind, payload = self._output_queue.get_nowait()
            except Empty:
                return
            if kind == "chunk":
                yield payload
            elif kind == "error":
                raise payload
            elif kind == "done":
                raise RuntimeError("Async codec decoder finished before close_and_drain().")

    def close_and_drain(self) -> Iterator[np.ndarray]:
        if not self._closed:
            self._closed = True
            started_at = time.perf_counter()
            self._input_queue.put(None)
            add_metric_time(self._metrics, "async_decode_input_queue_wait_s", time.perf_counter() - started_at)

        while True:
            kind, payload = self._output_queue.get()
            if kind == "chunk":
                yield payload
            elif kind == "error":
                self._thread.join(timeout=1.0)
                raise payload
            elif kind == "done":
                self._thread.join(timeout=1.0)
                add_metric_time(self._metrics, "async_decode_thread_join_s", 0.0)
                return

    def _worker(self) -> None:
        try:
            while True:
                audio_frames = self._input_queue.get()
                if audio_frames is None:
                    started_at = time.perf_counter()
                    for chunk in flush_decoder(self._decoder):
                        self._put_chunk(chunk)
                    add_metric_time(self._metrics, "async_decoder_flush_worker_s", time.perf_counter() - started_at)
                    self._output_queue.put(("done", None))
                    return

                started_at = time.perf_counter()
                for chunk in decode_audio_frames(
                    audio_frames,
                    self._decoder,
                    self._codebook_size,
                    self._audio_eos_token,
                ):
                    self._put_chunk(chunk)
                elapsed_s = time.perf_counter() - started_at
                add_metric_time(self._metrics, "async_audio_token_decode_worker_s", elapsed_s)
                add_metric_count(self._metrics, "async_audio_token_decode_worker_calls")
        except BaseException as exc:
            self._output_queue.put(("error", exc))

    def _put_chunk(self, chunk: np.ndarray) -> None:
        if self._metrics is not None:
            self._metrics["async_decode_output_chunks"] = self._metrics.get("async_decode_output_chunks", 0) + 1
        self._output_queue.put(("chunk", chunk))


def run_streaming_tts(
    args: argparse.Namespace,
    session: MossTTSRealtimeStreamingSession,
    codec,
    decoder: AudioStreamDecoder,
    text_deltas: Iterator[str],
    metrics: dict[str, float] | None = None,
    print_text: bool = True,
) -> Iterator[np.ndarray]:
    codebook_size = int(getattr(codec, "codebook_size", 1024))
    audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))

    if getattr(session, "codec", None) is not codec:
        session.codec = codec

    with codec.streaming(batch_size=1):
        if metrics is not None:
            metrics["stream_start_perf"] = time.perf_counter()
            metrics["async_codec_decode"] = False
        for delta in text_deltas:
            loop_started_at = time.perf_counter()
            if metrics is not None:
                metrics["text_delta_count"] = metrics.get("text_delta_count", 0) + 1
                metrics["text_chars_streamed"] = metrics.get("text_chars_streamed", 0) + len(delta)
            if print_text:
                print_started_at = time.perf_counter()
                print(delta, end="", flush=True)
                add_metric_time(metrics, "stdout_print_s", time.perf_counter() - print_started_at)
            sync_cuda_if_enabled(metrics)
            model_started_at = time.perf_counter()
            audio_frames = session.push_text(delta)
            sync_cuda_if_enabled(metrics)
            add_metric_time(metrics, "session_push_text_s", time.perf_counter() - model_started_at)
            add_metric_count(metrics, "session_push_text_calls")
            if metrics is not None:
                metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
                if audio_frames and "first_audio_tokens_s" not in metrics:
                    metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
            decode_started_at = time.perf_counter()
            yield from note_audio_chunks(
                decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token),
                metrics,
            )
            add_metric_time(metrics, "audio_token_decode_iteration_s", time.perf_counter() - decode_started_at)
            add_metric_time(metrics, "stream_delta_loop_s", time.perf_counter() - loop_started_at)

        sync_cuda_if_enabled(metrics)
        end_started_at = time.perf_counter()
        audio_frames = session.end_text()
        sync_cuda_if_enabled(metrics)
        add_metric_time(metrics, "session_end_text_s", time.perf_counter() - end_started_at)
        add_metric_count(metrics, "session_end_text_calls")
        if metrics is not None:
            metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
            if audio_frames and "first_audio_tokens_s" not in metrics:
                metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
        decode_started_at = time.perf_counter()
        yield from note_audio_chunks(
            decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token),
            metrics,
        )
        add_metric_time(metrics, "audio_token_decode_iteration_s", time.perf_counter() - decode_started_at)

        while True:
            sync_cuda_if_enabled(metrics)
            drain_started_at = time.perf_counter()
            drain_max_steps = current_drain_max_steps(args, metrics)
            audio_frames = session.drain(max_steps=drain_max_steps)
            sync_cuda_if_enabled(metrics)
            add_metric_time(metrics, "session_drain_s", time.perf_counter() - drain_started_at)
            add_metric_count(metrics, "session_drain_calls")
            add_metric_count(metrics, "session_drain_requested_steps", drain_max_steps)
            if not audio_frames:
                break
            if metrics is not None:
                metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
                if audio_frames and "first_audio_tokens_s" not in metrics:
                    metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
            decode_started_at = time.perf_counter()
            yield from note_audio_chunks(
                decode_audio_frames(audio_frames, decoder, codebook_size, audio_eos_token),
                metrics,
            )
            add_metric_time(metrics, "audio_token_decode_iteration_s", time.perf_counter() - decode_started_at)
            if session.inferencer.is_finished:
                break

        flush_started_at = time.perf_counter()
        yield from note_audio_chunks(flush_decoder(decoder), metrics)
        add_metric_time(metrics, "decoder_flush_s", time.perf_counter() - flush_started_at)


def run_streaming_tts_async_decode(
    args: argparse.Namespace,
    session: MossTTSRealtimeStreamingSession,
    codec,
    decoder: AudioStreamDecoder,
    text_deltas: Iterator[str],
    *,
    queue_size: int,
    metrics: dict[str, float] | None = None,
    print_text: bool = True,
) -> Iterator[np.ndarray]:
    codebook_size = int(getattr(codec, "codebook_size", 1024))
    audio_eos_token = int(getattr(session.inferencer, "audio_eos_token", 1026))

    with codec.streaming(batch_size=1):
        if metrics is not None:
            metrics["stream_start_perf"] = time.perf_counter()
            metrics["async_codec_decode"] = True
            metrics["async_codec_queue_size"] = max(1, queue_size)
        async_decoder = AsyncCodecDecoder(
            decoder,
            codebook_size,
            audio_eos_token,
            queue_size=queue_size,
            metrics=metrics,
        )

        def submit_and_yield(audio_frames: list[torch.Tensor]) -> Iterator[np.ndarray]:
            submit_started_at = time.perf_counter()
            async_decoder.submit(audio_frames)
            add_metric_time(metrics, "async_decode_submit_s", time.perf_counter() - submit_started_at)
            yield from note_audio_chunks(async_decoder.available_chunks(), metrics)

        try:
            for delta in text_deltas:
                yield from note_audio_chunks(async_decoder.available_chunks(), metrics)
                loop_started_at = time.perf_counter()
                if metrics is not None:
                    metrics["text_delta_count"] = metrics.get("text_delta_count", 0) + 1
                    metrics["text_chars_streamed"] = metrics.get("text_chars_streamed", 0) + len(delta)
                if print_text:
                    print_started_at = time.perf_counter()
                    print(delta, end="", flush=True)
                    add_metric_time(metrics, "stdout_print_s", time.perf_counter() - print_started_at)
                sync_cuda_if_enabled(metrics)
                model_started_at = time.perf_counter()
                audio_frames = session.push_text(delta)
                sync_cuda_if_enabled(metrics)
                add_metric_time(metrics, "session_push_text_s", time.perf_counter() - model_started_at)
                add_metric_count(metrics, "session_push_text_calls")
                if metrics is not None:
                    metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
                    if audio_frames and "first_audio_tokens_s" not in metrics:
                        metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
                yield from submit_and_yield(audio_frames)
                add_metric_time(metrics, "stream_delta_loop_s", time.perf_counter() - loop_started_at)

            sync_cuda_if_enabled(metrics)
            end_started_at = time.perf_counter()
            audio_frames = session.end_text()
            sync_cuda_if_enabled(metrics)
            add_metric_time(metrics, "session_end_text_s", time.perf_counter() - end_started_at)
            add_metric_count(metrics, "session_end_text_calls")
            if metrics is not None:
                metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
                if audio_frames and "first_audio_tokens_s" not in metrics:
                    metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
            yield from submit_and_yield(audio_frames)

            while True:
                yield from note_audio_chunks(async_decoder.available_chunks(), metrics)
                sync_cuda_if_enabled(metrics)
                drain_started_at = time.perf_counter()
                drain_max_steps = current_drain_max_steps(args, metrics)
                audio_frames = session.drain(max_steps=drain_max_steps)
                sync_cuda_if_enabled(metrics)
                add_metric_time(metrics, "session_drain_s", time.perf_counter() - drain_started_at)
                add_metric_count(metrics, "session_drain_calls")
                add_metric_count(metrics, "session_drain_requested_steps", drain_max_steps)
                if not audio_frames:
                    break
                if metrics is not None:
                    metrics["model_audio_frame_batches"] = metrics.get("model_audio_frame_batches", 0) + len(audio_frames)
                    if audio_frames and "first_audio_tokens_s" not in metrics:
                        metrics["first_audio_tokens_s"] = time.perf_counter() - metrics["stream_start_perf"]
                yield from submit_and_yield(audio_frames)
                if session.inferencer.is_finished:
                    break

            yield from note_audio_chunks(async_decoder.close_and_drain(), metrics)
        except BaseException:
            if not async_decoder._closed:
                async_decoder._closed = True
                async_decoder._input_queue.put(None)
            raise


def consume_chunks(chunks: Iterator[np.ndarray]) -> tuple[int, float]:
    sample_count = 0
    started_at = time.perf_counter()
    for chunk in chunks:
        sample_count += int(chunk.reshape(-1).shape[0])
    return sample_count, time.perf_counter() - started_at


def reset_session_for_generation(
    session: MossTTSRealtimeStreamingSession,
    input_ids: np.ndarray,
) -> None:
    session.reset_turn(
        input_ids=input_ids,
        include_system_prompt=False,
        reset_cache=True,
    )


def make_text_deltas_from_text(args: argparse.Namespace, text: str) -> Iterator[str]:
    return chunk_text(text, args.delta_chunk_chars, args.delta_delay_s)


def write_wav_with_metrics(
    out_path: Path,
    sample_rate: int,
    chunks: Iterator[np.ndarray],
    metrics: dict[str, float] | None = None,
) -> None:
    collect_started_at = time.perf_counter()
    all_chunks: list[np.ndarray] = []
    sample_count = 0
    for chunk in chunks:
        chunk = chunk.astype(np.float32).reshape(-1)
        sample_count += int(chunk.shape[0])
        all_chunks.append(chunk)
    stream_end_perf = time.perf_counter()

    if not all_chunks:
        raise RuntimeError("No audio chunks produced.")

    prepare_started_at = time.perf_counter()
    audio = np.concatenate(all_chunks)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    prepare_s = time.perf_counter() - prepare_started_at

    out_path.parent.mkdir(parents=True, exist_ok=True)
    file_write_started_at = time.perf_counter()
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    file_write_s = time.perf_counter() - file_write_started_at

    if metrics is not None:
        stream_total_s = stream_end_perf - metrics["stream_start_perf"]
        audio_duration_s = sample_count / float(sample_rate)
        intervals = metrics.pop("_audio_chunk_intervals_s", [])
        metrics["chunk_collection_s"] = stream_end_perf - collect_started_at
        metrics["audio_array_prepare_s"] = prepare_s
        metrics["file_write_s"] = file_write_s
        metrics["write_wav_s"] = prepare_s + file_write_s
        metrics["audio_samples"] = sample_count
        metrics["audio_duration_s"] = audio_duration_s
        metrics["stream_total_s"] = stream_total_s
        metrics["rtf_stream"] = stream_total_s / audio_duration_s if audio_duration_s else None
        metrics["audio_seconds_per_wall_second"] = audio_duration_s / stream_total_s if stream_total_s else None
        for key in (
            "session_push_text_s",
            "session_end_text_s",
            "session_drain_s",
            "codec_decode_s",
            "audio_token_decode_iteration_s",
            "async_audio_token_decode_worker_s",
            "async_decode_input_queue_wait_s",
            "async_decode_submit_s",
            "stdout_print_s",
            "write_wav_s",
        ):
            if key in metrics and stream_total_s:
                metrics[f"{key}_pct_stream"] = metrics[key] / stream_total_s
        model_total_s = (
            metrics.get("session_push_text_s", 0.0)
            + metrics.get("session_end_text_s", 0.0)
            + metrics.get("session_drain_s", 0.0)
        )
        metrics["model_stream_calls_s"] = model_total_s
        metrics["model_stream_calls_pct_stream"] = model_total_s / stream_total_s if stream_total_s else None
        if metrics.get("audio_chunk_count"):
            metrics["mean_audio_chunk_duration_s"] = audio_duration_s / metrics["audio_chunk_count"]
            metrics["largest_audio_chunk_duration_s"] = metrics.get("largest_audio_chunk_samples", 0) / float(sample_rate)
        if intervals:
            intervals_arr = np.asarray(intervals, dtype=np.float64)
            metrics["chunk_interval_mean_s"] = float(intervals_arr.mean())
            metrics["chunk_interval_p50_s"] = float(np.percentile(intervals_arr, 50))
            metrics["chunk_interval_p95_s"] = float(np.percentile(intervals_arr, 95))
            metrics["chunk_interval_max_s"] = float(intervals_arr.max())
        if "first_audio_chunk_s" in metrics:
            after_ttfb_s = max(0.0, stream_total_s - metrics["first_audio_chunk_s"])
            remaining_audio_s = max(0.0, audio_duration_s)
            metrics["rtf_after_first_chunk"] = after_ttfb_s / remaining_audio_s if remaining_audio_s else None
            metrics["audio_seconds_per_wall_second_after_first_chunk"] = (
                remaining_audio_s / after_ttfb_s if after_ttfb_s else None
            )


def benchmark_path_for(out_path: Path, benchmark_json: str | None) -> Path:
    if benchmark_json:
        path = Path(benchmark_json).expanduser()
        return path if path.is_absolute() else ROOT_DIR / path
    return out_path.with_suffix(out_path.suffix + ".benchmark.json")


def write_benchmark(metrics: dict[str, object], out_path: Path, args: argparse.Namespace) -> None:
    bench_path = benchmark_path_for(out_path, args.benchmark_json)
    persisted = {
        key: value
        for key, value in metrics.items()
        if not key.endswith("_perf")
    }
    bench_path.parent.mkdir(parents=True, exist_ok=True)
    bench_path.write_text(json.dumps(persisted, indent=2, sort_keys=True) + "\n")

    print("\n[BENCHMARK]")
    for key in (
        "setup_total_s",
        "tokenizer_load_s",
        "model_load_s",
        "codec_load_s",
        "prompt_encode_s",
        "session_init_s",
        "first_audio_tokens_s",
        "first_audio_chunk_s",
        "stream_total_s",
        "audio_duration_s",
        "rtf_stream",
        "audio_seconds_per_wall_second",
        "rtf_after_first_chunk",
        "audio_seconds_per_wall_second_after_first_chunk",
        "model_stream_calls_s",
        "model_stream_calls_pct_stream",
        "runtime_profile",
        "session_drain_calls",
        "session_drain_requested_steps",
        "drain_max_steps",
        "first_drain_max_steps",
        "async_codec_decode",
        "async_codec_queue_size",
        "codec_decode_s",
        "codec_decode_s_pct_stream",
        "audio_token_decode_iteration_s",
        "audio_token_decode_iteration_s_pct_stream",
        "async_audio_token_decode_worker_s",
        "async_audio_token_decode_worker_s_pct_stream",
        "async_audio_token_decode_worker_calls",
        "async_decode_input_queue_wait_s",
        "async_decode_input_queue_wait_s_pct_stream",
        "async_decode_submit_s",
        "async_decode_submit_s_pct_stream",
        "async_decode_submitted_batches",
        "async_decode_submitted_frames",
        "async_decode_output_chunks",
        "write_wav_s",
        "chunk_interval_p50_s",
        "chunk_interval_p95_s",
        "chunk_interval_max_s",
        "mean_audio_chunk_duration_s",
        "audio_chunk_count",
        "model_audio_frame_batches",
        "text_delta_count",
    ):
        if key in persisted:
            print(f"{key}: {persisted[key]}")
    print(f"benchmark_json: {bench_path}")


def main() -> None:
    args = parse_args()
    process_started_at = time.perf_counter()
    metrics: dict[str, object] | None = {} if args.benchmark else None

    configure_torch_runtime(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this demo.")

    prompt_wav = Path(args.prompt_wav)
    if not prompt_wav.is_absolute():
        prompt_wav = ROOT_DIR / prompt_wav
    args.prompt_wav = str(prompt_wav)
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt WAV not found: {prompt_wav}")

    device = torch.device(args.device)
    out_path = make_output_path(args.out_wav)

    if args.warmup_runs > 0 and args.text is None:
        raise ValueError("--warmup-runs requires --text because stdin cannot be replayed for warmup and measurement.")

    if args.text is not None:
        text_deltas = chunk_text(args.text, args.delta_chunk_chars, args.delta_delay_s)
    else:
        if sys.stdin.isatty():
            print("Reading text from stdin. Press Ctrl-D when finished.", file=sys.stderr)
        text_deltas = stdin_chunks(args.delta_chunk_chars, args.delta_delay_s)

    print(f"[INFO] device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[INFO] prompt_wav={prompt_wav}")
    print(f"[INFO] out_wav={out_path}")
    if metrics is not None:
        metrics.update(
            {
                "model_path": args.model_path,
                "codec_path": args.codec_path,
                "runtime_profile": args.runtime_profile,
                "prompt_wav": str(prompt_wav),
                "out_wav": str(out_path),
                "device": str(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "attn_implementation": args.attn_implementation,
                "delta_chunk_chars": args.delta_chunk_chars,
                "delta_delay_s": args.delta_delay_s,
                "decode_chunk_frames": args.decode_chunk_frames,
                "decode_overlap_frames": args.decode_overlap_frames,
                "drain_max_steps": args.drain_max_steps,
                "first_drain_max_steps": args.first_drain_max_steps,
                "async_codec_decode": args.async_codec_decode,
                "async_codec_queue_size": args.async_codec_queue_size,
                "codec_chunk_duration": args.codec_chunk_duration,
                "sample_rate": args.sample_rate,
                "max_length": args.max_length,
                "no_sample": args.no_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "repetition_window": args.repetition_window,
                "local_compile": args.local_compile,
                "local_compile_backend": args.local_compile_backend,
                "local_compile_mode": args.local_compile_mode,
                "local_compile_fullgraph": args.local_compile_fullgraph,
                "local_compile_dynamic": args.local_compile_dynamic,
                "dynamo_cache_size_limit": args.dynamo_cache_size_limit,
                "benchmark_synchronize": args.benchmark_synchronize,
                "warmup_runs": args.warmup_runs,
                "allow_tf32": args.allow_tf32,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
        )
    session, codec, decoder, input_ids = build_session(args, device, metrics=metrics)

    if args.warmup_runs > 0:
        warmup_records = []
        for warmup_idx in range(args.warmup_runs):
            print(f"[INFO] warmup_run={warmup_idx + 1}/{args.warmup_runs}", file=sys.stderr)
            reset_session_for_generation(session, input_ids)
            warmup_decoder = create_decoder(args, codec, device)
            if args.async_codec_decode:
                warmup_chunks = run_streaming_tts_async_decode(
                    args,
                    session,
                    codec,
                    warmup_decoder,
                    make_text_deltas_from_text(args, args.text),
                    queue_size=args.async_codec_queue_size,
                    metrics=None,
                    print_text=False,
                )
            else:
                warmup_chunks = run_streaming_tts(
                    args,
                    session,
                    codec,
                    warmup_decoder,
                    make_text_deltas_from_text(args, args.text),
                    metrics=None,
                    print_text=False,
                )
            sample_count, elapsed_s = consume_chunks(warmup_chunks)
            warmup_records.append(
                {
                    "run": warmup_idx + 1,
                    "elapsed_s": elapsed_s,
                    "audio_duration_s": sample_count / float(args.sample_rate),
                    "audio_samples": sample_count,
                }
            )
        if metrics is not None:
            metrics["warmup_records"] = warmup_records
        reset_session_for_generation(session, input_ids)
        decoder = create_decoder(args, codec, device)
        text_deltas = make_text_deltas_from_text(args, args.text)

    if args.async_codec_decode:
        wav_chunks = run_streaming_tts_async_decode(
            args,
            session,
            codec,
            decoder,
            text_deltas,
            queue_size=args.async_codec_queue_size,
            metrics=metrics,
        )
    else:
        wav_chunks = run_streaming_tts(args, session, codec, decoder, text_deltas, metrics=metrics)
    if metrics is not None:
        write_wav_with_metrics(out_path, args.sample_rate, wav_chunks, metrics=metrics)
        metrics["process_total_s"] = time.perf_counter() - process_started_at
        write_benchmark(metrics, out_path, args)
    else:
        write_wav(out_path, args.sample_rate, wav_chunks)
    print(f"\n[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
