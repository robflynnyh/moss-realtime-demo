#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BOOT_ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("UV_CACHE_DIR", str(_BOOT_ROOT_DIR / ".uv-cache"))
os.environ.setdefault("PIP_CACHE_DIR", str(_BOOT_ROOT_DIR / ".uv-cache" / "pip"))
os.environ.setdefault("XDG_CACHE_HOME", str(_BOOT_ROOT_DIR / ".uv-cache" / "xdg"))
os.environ.setdefault("HF_HOME", str(_BOOT_ROOT_DIR / ".hf-cache"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_BOOT_ROOT_DIR / ".hf-cache" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_BOOT_ROOT_DIR / ".hf-cache" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(_BOOT_ROOT_DIR / ".uv-cache" / "torch"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_BOOT_ROOT_DIR / ".uv-cache" / "triton"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(_BOOT_ROOT_DIR / ".uv-cache" / "vllm"))
os.environ.setdefault("VLLM_CONFIG_ROOT", str(_BOOT_ROOT_DIR / ".uv-cache" / "vllm-config"))
os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", str(_BOOT_ROOT_DIR / ".uv-cache" / "flashinfer-workspace"))

import numpy as np
import torch
from transformers import AutoTokenizer

from moss_streaming_demo import (
    ROOT_DIR,
    SAMPLE_RATE,
    _extract_codes,
    _load_audio,
    configure_local_compile,
    configure_torch_runtime,
    load_codec,
    sync_cuda_if_enabled,
)
from moss_streaming_demo import MossTTSRealtime, MossTTSRealtimeInference, MossTTSRealtimeProcessor


@dataclass(frozen=True)
class TextItem:
    idx: int
    item_id: str
    text: str
    prompt_wav: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MOSS-TTS-Realtime rollout audio in microbatches and save WAVs."
    )
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-TTS-Realtime")
    parser.add_argument("--codec-path", default="OpenMOSS-Team/MOSS-Audio-Tokenizer")
    parser.add_argument("--prompt-wav", default=str(ROOT_DIR / "prompts" / "nabu_joe_en_us_12s.wav"))
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Rollout text. May be passed multiple times.",
    )
    parser.add_argument(
        "--texts-file",
        default=None,
        help="Plain text file with one rollout per line, or JSONL with text plus optional id fields.",
    )
    parser.add_argument("--out-dir", default=None, help="Defaults to outputs/batch_rollout_<timestamp>.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--packing-mode",
        choices=["interleaved", "full-text"],
        default="interleaved",
        help="interleaved matches streaming deployment: prefill a short text prefix, then step next text token plus previous audio. full-text prefills all text before audio drain.",
    )
    parser.add_argument(
        "--prefill-text-len",
        type=int,
        default=None,
        help="Initial text tokens per sample for interleaved mode. Defaults to the processor delay length.",
    )
    parser.add_argument(
        "--max-audio-steps",
        type=int,
        default=512,
        help="Maximum autoregressive audio-token steps after prefill per microbatch.",
    )
    parser.add_argument("--max-length", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--codec-chunk-duration", type=float, default=0.24)
    parser.add_argument(
        "--codec-decode-batch-size",
        type=int,
        default=16,
        help="Decode waveform in subbatches after audio-token generation. 0 decodes the full microbatch at once.",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--repetition-window", type=int, default=50)
    parser.add_argument("--no-sample", dest="no_sample", action="store_true", default=True)
    parser.add_argument("--sample", dest="no_sample", action="store_false")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Print and write stage timing metrics.")
    parser.add_argument(
        "--benchmark-json",
        default=None,
        help="Benchmark JSON path. Defaults to <out-dir>/benchmark.json when --benchmark is set.",
    )
    parser.add_argument(
        "--benchmark-synchronize",
        action="store_true",
        help="Synchronize CUDA around measured GPU stages for cleaner attribution.",
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
        help="Enable torch.compile on the local transformer. Off by default on this hardware.",
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
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.max_audio_steps < 0:
        raise ValueError("--max-audio-steps must be >= 0.")
    if args.prefill_text_len is not None and args.prefill_text_len < 1:
        raise ValueError("--prefill-text-len must be >= 1.")
    if args.codec_decode_batch_size < 0:
        raise ValueError("--codec-decode-batch-size must be >= 0.")
    if args.top_k is not None and args.top_k <= 0:
        args.top_k = None
    if args.top_p is not None and args.top_p >= 1.0:
        args.top_p = None
    return args


def resolve_repo_path(path_like: str | None, default: Path) -> Path:
    if path_like is None:
        return default
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def default_out_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT_DIR / "outputs" / f"batch_rollout_{stamp}"


def safe_stem(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    return cleaned[:96] if cleaned else fallback


def read_text_items(args: argparse.Namespace) -> list[TextItem]:
    items: list[TextItem] = []
    for text in args.text:
        text = str(text)
        if text.strip():
            items.append(TextItem(len(items), f"text_{len(items):04d}", text))

    if args.texts_file:
        path = resolve_repo_path(args.texts_file, ROOT_DIR / args.texts_file)
        if not path.exists():
            raise FileNotFoundError(f"Texts file not found: {path}")
        is_jsonl = path.suffix.lower() == ".jsonl"
        for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            if is_jsonl:
                record = json.loads(line)
                if "text" not in record:
                    raise ValueError(f"{path}:{line_no} is missing a text field.")
                text = str(record["text"])
                item_id = str(record.get("id") or record.get("uid") or f"line_{line_no:04d}")
                prompt_wav = record.get("prompt_wav")
                item_idx = int(record.get("idx", len(items)))
            else:
                text = raw_line
                item_id = f"line_{line_no:04d}"
                prompt_wav = None
                item_idx = len(items)
            if text.strip():
                prompt_path = None
                if prompt_wav:
                    prompt_path = resolve_repo_path(str(prompt_wav), ROOT_DIR / str(prompt_wav))
                items.append(TextItem(item_idx, item_id, text, prompt_path))

    if not items:
        raise ValueError("Pass at least one --text or --texts-file item.")
    return items


def resolve_item_prompt_wavs(items: list[TextItem], default_prompt_wav: Path) -> list[TextItem]:
    resolved = []
    for item in items:
        prompt_wav = item.prompt_wav or default_prompt_wav
        if not prompt_wav.exists():
            raise FileNotFoundError(f"Prompt WAV not found for item {item.item_id}: {prompt_wav}")
        resolved.append(TextItem(item.idx, item.item_id, item.text, prompt_wav.resolve()))
    return resolved


def batched(items: list[TextItem], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def batched_by_prompt(items: list[TextItem], batch_size: int):
    groups: dict[Path, list[TextItem]] = {}
    for item in items:
        if item.prompt_wav is None:
            raise ValueError(f"Internal error: item {item.item_id} has no resolved prompt_wav")
        groups.setdefault(item.prompt_wav, []).append(item)
    for prompt_wav, group_items in groups.items():
        for batch_items in batched(group_items, batch_size):
            yield prompt_wav, batch_items


def write_pcm16_wav(path: Path, sample_rate: int, audio: torch.Tensor) -> int:
    wav = audio.detach().float().cpu().reshape(-1).numpy()
    wav = np.clip(wav, -1.0, 1.0)
    pcm16 = (wav * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return int(wav.shape[0])


def build_rollout_prefix(processor, tokenizer, prompt_tokens: np.ndarray) -> np.ndarray:
    system_prompt = processor.make_ensemble(prompt_tokens)
    assistant_prefix_ids = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n")
    assistant_prefix = np.full(
        (len(assistant_prefix_ids), system_prompt.shape[1]),
        fill_value=processor.audio_channel_pad,
        dtype=np.int64,
    )
    assistant_prefix[:, 0] = assistant_prefix_ids
    return np.concatenate([system_prompt, assistant_prefix], axis=0)


def encode_prompt_prefix(
    args: argparse.Namespace,
    processor,
    tokenizer,
    codec,
    prompt_wav: Path,
    device: torch.device,
    metrics: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    prompt_metrics: dict[str, Any] = {"prompt_wav": str(prompt_wav)}
    with torch.inference_mode():
        started = time.perf_counter()
        prompt_audio = _load_audio(prompt_wav, target_sample_rate=args.sample_rate)
        prompt_metrics["prompt_audio_load_s"] = time.perf_counter() - started
        prompt_metrics["prompt_audio_duration_s"] = float(prompt_audio.shape[-1]) / float(args.sample_rate)

        sync_cuda_if_enabled(metrics)
        started = time.perf_counter()
        prompt_result = codec.encode(
            prompt_audio.unsqueeze(0).to(device),
            chunk_duration=args.codec_chunk_duration,
        )
        prompt_tokens = _extract_codes(prompt_result).cpu().numpy().squeeze(1)
        sync_cuda_if_enabled(metrics)
        prompt_metrics["prompt_encode_s"] = time.perf_counter() - started
        prompt_metrics["prompt_token_shape"] = list(prompt_tokens.shape)

    started = time.perf_counter()
    prefix_input_ids = build_rollout_prefix(processor, tokenizer, prompt_tokens)
    prompt_metrics["prefix_build_s"] = time.perf_counter() - started
    prompt_metrics["prefix_shape"] = list(prefix_input_ids.shape)
    return prefix_input_ids, prompt_metrics


def load_stack(
    args: argparse.Namespace,
    device: torch.device,
    metrics: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    setup_started = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metrics["dtype"] = str(dtype).replace("torch.", "")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    metrics["tokenizer_load_s"] = time.perf_counter() - started

    started = time.perf_counter()
    processor = MossTTSRealtimeProcessor(tokenizer)
    metrics["processor_init_s"] = time.perf_counter() - started

    started = time.perf_counter()
    model = MossTTSRealtime.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    metrics["model_load_s"] = time.perf_counter() - started

    started = time.perf_counter()
    codec = load_codec(args.codec_path, device)
    metrics["codec_load_s"] = time.perf_counter() - started

    inferencer = MossTTSRealtimeInference(model, tokenizer, max_length=args.max_length)
    configure_local_compile(inferencer, args, metrics=metrics)
    metrics["setup_total_s"] = time.perf_counter() - setup_started
    return tokenizer, processor, model, codec, inferencer


def append_valid_tokens(
    batch_tokens: torch.Tensor,
    per_sample_tokens: list[list[torch.Tensor]],
    stopped: list[bool],
    *,
    codebook_size: int,
    audio_eos_token: int,
) -> int:
    appended = 0
    for batch_idx in range(batch_tokens.shape[0]):
        if stopped[batch_idx]:
            continue
        row = batch_tokens[batch_idx].detach()
        first_code = int(row[0].item())
        invalid = bool(((row < 0) | (row >= codebook_size)).any().item())
        if first_code == audio_eos_token or invalid:
            stopped[batch_idx] = True
            continue
        per_sample_tokens[batch_idx].append(row.clone())
        appended += 1
    return appended


def decode_batch(
    codec,
    codes_list: list[torch.Tensor],
    metrics: dict[str, Any],
    *,
    codec_decode_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sync_cuda_if_enabled(metrics)
    started = time.perf_counter()
    decode_batch_size = int(codec_decode_batch_size)
    if decode_batch_size <= 0 or decode_batch_size >= len(codes_list):
        decoded = codec.batch_decode(codes_list)
        sync_cuda_if_enabled(metrics)
        metrics["codec_batch_decode_s"] = metrics.get("codec_batch_decode_s", 0.0) + (time.perf_counter() - started)
        metrics["codec_batch_decode_calls"] = metrics.get("codec_batch_decode_calls", 0) + 1
        metrics["codec_decode_subbatch_count"] = metrics.get("codec_decode_subbatch_count", 0) + 1
        metrics["codec_decode_effective_batch_size"] = len(codes_list)
        audio = decoded["audio"] if isinstance(decoded, dict) else decoded.audio
        lengths = decoded["audio_lengths"] if isinstance(decoded, dict) else decoded.audio_lengths
        return audio.detach().cpu(), lengths.detach().cpu()

    audio_chunks = []
    length_chunks = []
    subbatch_count = 0
    largest_subbatch = 0
    for start_idx in range(0, len(codes_list), decode_batch_size):
        sub_codes = codes_list[start_idx : start_idx + decode_batch_size]
        subbatch_count += 1
        largest_subbatch = max(largest_subbatch, len(sub_codes))
        decoded = codec.batch_decode(sub_codes)
        sync_cuda_if_enabled(metrics)
        audio = decoded["audio"] if isinstance(decoded, dict) else decoded.audio
        lengths = decoded["audio_lengths"] if isinstance(decoded, dict) else decoded.audio_lengths
        audio_chunks.append(audio.detach().cpu())
        length_chunks.append(lengths.detach().cpu())
        del decoded, audio, lengths
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics["codec_batch_decode_s"] = metrics.get("codec_batch_decode_s", 0.0) + (time.perf_counter() - started)
    metrics["codec_batch_decode_calls"] = metrics.get("codec_batch_decode_calls", 0) + subbatch_count
    metrics["codec_decode_subbatch_count"] = metrics.get("codec_decode_subbatch_count", 0) + subbatch_count
    metrics["codec_decode_effective_batch_size"] = largest_subbatch
    max_audio_len = max(chunk.shape[-1] for chunk in audio_chunks)
    padded_audio_chunks = []
    for chunk in audio_chunks:
        if chunk.shape[-1] == max_audio_len:
            padded_audio_chunks.append(chunk)
            continue
        pad_shape = list(chunk.shape)
        pad_shape[-1] = max_audio_len - chunk.shape[-1]
        padding = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
        padded_audio_chunks.append(torch.cat([chunk, padding], dim=-1))
    return torch.cat(padded_audio_chunks, dim=0), torch.cat(length_chunks, dim=0)


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
    }


def cuda_peak_memory_for_stage(device: torch.device, stage: str) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        f"{stage}_cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        f"{stage}_cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        f"{stage}_cuda_current_allocated_mib": torch.cuda.memory_allocated(device) / (1024**2),
        f"{stage}_cuda_current_reserved_mib": torch.cuda.memory_reserved(device) / (1024**2),
    }


def initial_text_prefixes(
    tokenized: list[list[int]],
    *,
    packing_mode: str,
    prefill_text_len: int,
) -> tuple[list[list[int]], list[int]]:
    if packing_mode == "full-text":
        return [list(ids) for ids in tokenized], [len(ids) for ids in tokenized]

    prefixes = []
    cursors = []
    for ids in tokenized:
        prefix_len = min(len(ids), prefill_text_len)
        prefixes.append(list(ids[:prefix_len]))
        cursors.append(prefix_len)
    return prefixes, cursors


def next_interleaved_text_tokens(
    tokenized: list[list[int]],
    cursors: list[int],
    stopped: list[bool],
    *,
    text_pad_id: int,
) -> tuple[list[int], int]:
    text_tokens = []
    real_token_count = 0
    for batch_idx, ids in enumerate(tokenized):
        if stopped[batch_idx] or cursors[batch_idx] >= len(ids):
            text_tokens.append(text_pad_id)
            continue
        text_tokens.append(int(ids[cursors[batch_idx]]))
        cursors[batch_idx] += 1
        real_token_count += 1
    return text_tokens, real_token_count


def generate_microbatch(
    args: argparse.Namespace,
    tokenizer,
    prefill_text_len: int,
    inferencer: MossTTSRealtimeInference,
    codec,
    prefix_input_ids: np.ndarray,
    items: list[TextItem],
    out_dir: Path,
    batch_index: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    batch_started = time.perf_counter()
    device = inferencer.device
    reset_cuda_peak(device)
    tokenized = [tokenizer.encode(item.text, add_special_tokens=False) for item in items]
    empty = [item.item_id for item, ids in zip(items, tokenized) if not ids]
    if empty:
        raise ValueError(f"Empty tokenized text for item ids: {empty}")

    inferencer.reset_generation_state(keep_cache=False)
    per_sample_tokens: list[list[torch.Tensor]] = [[] for _ in items]
    stopped = [False for _ in items]
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": not args.no_sample,
        "repetition_penalty": args.repetition_penalty,
        "repetition_window": args.repetition_window if args.repetition_window > 0 else None,
    }
    codebook_size = int(getattr(codec, "codebook_size", 1024))
    audio_eos_token = int(getattr(inferencer, "audio_eos_token", 1026))
    text_pad_id = int(getattr(inferencer, "text_pad_id", 151655))
    text_prefixes, text_cursors = initial_text_prefixes(
        tokenized,
        packing_mode=args.packing_mode,
        prefill_text_len=prefill_text_len,
    )
    post_prefill_cursors = list(text_cursors)

    sync_cuda_if_enabled(metrics)
    reset_cuda_peak(device)
    started = time.perf_counter()
    prefill_kwargs = dict(generation_kwargs)
    prefill_kwargs["repetition_penalty"] = None
    first_tokens = inferencer.prefill(
        input_ids=[prefix_input_ids] * len(items),
        text_prefix_ids=text_prefixes,
        **prefill_kwargs,
    )
    sync_cuda_if_enabled(metrics)
    prefill_s = time.perf_counter() - started
    prefill_memory = cuda_peak_memory_for_stage(device, "prefill")
    append_valid_tokens(
        first_tokens,
        per_sample_tokens,
        stopped,
        codebook_size=codebook_size,
        audio_eos_token=audio_eos_token,
    )

    sync_cuda_if_enabled(metrics)
    reset_cuda_peak(device)
    started = time.perf_counter()
    step_count = 0
    appended_count = 0
    text_step_calls = 0
    drain_step_calls = 0
    text_tokens_stepped = 0
    while step_count < args.max_audio_steps and not all(stopped) and not inferencer.is_finished:
        if args.packing_mode == "interleaved":
            text_arg, real_token_count = next_interleaved_text_tokens(
                tokenized,
                text_cursors,
                stopped,
                text_pad_id=text_pad_id,
            )
            if real_token_count:
                text_step_calls += 1
                text_tokens_stepped += real_token_count
            else:
                drain_step_calls += 1
        else:
            text_arg = None
            drain_step_calls += 1
        tokens = inferencer.step(text_arg, **generation_kwargs)
        appended_count += append_valid_tokens(
            tokens,
            per_sample_tokens,
            stopped,
            codebook_size=codebook_size,
            audio_eos_token=audio_eos_token,
        )
        step_count += 1
    sync_cuda_if_enabled(metrics)
    generate_s = time.perf_counter() - started
    generate_memory = cuda_peak_memory_for_stage(device, "generate")

    if not any(per_sample_tokens):
        raise RuntimeError("No valid audio tokens generated for this microbatch.")

    codes_list: list[torch.Tensor] = []
    token_frames: list[int] = []
    for item, rows in zip(items, per_sample_tokens):
        if not rows:
            raise RuntimeError(f"No valid audio tokens generated for item {item.item_id}.")
        codes = torch.stack(rows, dim=0).transpose(0, 1).contiguous()
        codes_list.append(codes)
        token_frames.append(int(codes.shape[-1]))

    sync_cuda_if_enabled(metrics)
    reset_cuda_peak(device)
    audio, lengths = decode_batch(
        codec,
        codes_list,
        metrics,
        codec_decode_batch_size=args.codec_decode_batch_size,
    )
    sync_cuda_if_enabled(metrics)
    decode_memory = cuda_peak_memory_for_stage(device, "codec_decode")

    write_started = time.perf_counter()
    records = []
    total_audio_samples = 0
    for local_idx, item in enumerate(items):
        stem = f"{item.idx:04d}_{safe_stem(item.item_id, f'item_{item.idx:04d}')}"
        wav_path = out_dir / f"{stem}.wav"
        length = int(lengths[local_idx].item())
        sample_count = write_pcm16_wav(wav_path, args.sample_rate, audio[local_idx, :, :length])
        total_audio_samples += sample_count
        try:
            path_for_record = str(wav_path.relative_to(ROOT_DIR))
        except ValueError:
            path_for_record = str(wav_path)
        records.append(
            {
                "idx": item.idx,
                "id": item.item_id,
                "path": path_for_record,
                "prompt_wav": str(item.prompt_wav) if item.prompt_wav else None,
                "text_chars": len(item.text),
                "text_tokens": len(tokenized[local_idx]),
                "audio_token_frames": token_frames[local_idx],
                "audio_samples": sample_count,
                "audio_duration_s": sample_count / float(args.sample_rate),
                "stopped": stopped[local_idx],
            }
        )
    write_s = time.perf_counter() - write_started

    batch_total_s = time.perf_counter() - batch_started
    audio_duration_s = total_audio_samples / float(args.sample_rate)
    batch_record = {
        "batch_index": batch_index,
        "batch_size": len(items),
        "prefill_s": prefill_s,
        "generate_s": generate_s,
        "write_wav_s": write_s,
        "batch_total_s": batch_total_s,
        "audio_duration_s": audio_duration_s,
        "audio_seconds_per_wall_second": audio_duration_s / batch_total_s if batch_total_s else None,
        "audio_seconds_per_generate_second": audio_duration_s / generate_s if generate_s else None,
        "packing_mode": args.packing_mode,
        "prefill_text_len": prefill_text_len,
        "prefill_text_tokens": sum(len(ids) for ids in text_prefixes),
        "total_text_tokens": sum(len(ids) for ids in tokenized),
        "remaining_text_tokens_after_prefill": sum(
            max(0, len(ids) - cursor) for ids, cursor in zip(tokenized, post_prefill_cursors)
        ),
        "remaining_text_tokens_after_generation": sum(
            max(0, len(ids) - cursor) for ids, cursor in zip(tokenized, text_cursors)
        ),
        "text_tokens_stepped": text_tokens_stepped,
        "prefill_text_tokens_per_s": sum(len(ids) for ids in text_prefixes) / prefill_s if prefill_s else None,
        "audio_step_calls": step_count,
        "text_step_calls": text_step_calls,
        "drain_step_calls": drain_step_calls,
        "audio_token_frames_appended_after_prefill": appended_count,
        "codec_decode_batch_size": args.codec_decode_batch_size,
        "codec_decode_effective_batch_size": (
            len(codes_list)
            if args.codec_decode_batch_size <= 0
            else min(args.codec_decode_batch_size, len(codes_list))
        ),
        "max_audio_steps": args.max_audio_steps,
        "all_stopped": all(stopped),
        "items": records,
    }
    batch_record.update(prefill_memory)
    batch_record.update(generate_memory)
    batch_record.update(decode_memory)
    stage_alloc_peaks = [
        batch_record.get("prefill_cuda_peak_allocated_mib"),
        batch_record.get("generate_cuda_peak_allocated_mib"),
        batch_record.get("codec_decode_cuda_peak_allocated_mib"),
    ]
    stage_reserved_peaks = [
        batch_record.get("prefill_cuda_peak_reserved_mib"),
        batch_record.get("generate_cuda_peak_reserved_mib"),
        batch_record.get("codec_decode_cuda_peak_reserved_mib"),
    ]
    batch_record["cuda_peak_allocated_mib"] = max((x for x in stage_alloc_peaks if x is not None), default=None)
    batch_record["cuda_peak_reserved_mib"] = max((x for x in stage_reserved_peaks if x is not None), default=None)
    return batch_record


def benchmark_path(out_dir: Path, benchmark_json: str | None) -> Path:
    if benchmark_json:
        path = Path(benchmark_json).expanduser()
        return path if path.is_absolute() else ROOT_DIR / path
    return out_dir / "benchmark.json"


def main() -> None:
    args = parse_args()
    process_started = time.perf_counter()
    configure_torch_runtime(args)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for batched rollout generation.")

    prompt_wav = resolve_repo_path(args.prompt_wav, ROOT_DIR / "prompts" / "nabu_joe_en_us_12s.wav")
    if not prompt_wav.exists():
        raise FileNotFoundError(f"Prompt WAV not found: {prompt_wav}")
    args.prompt_wav = str(prompt_wav)

    out_dir = resolve_repo_path(args.out_dir, default_out_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    items = resolve_item_prompt_wavs(read_text_items(args), prompt_wav.resolve())
    unique_prompt_wavs = sorted(
        {item.prompt_wav for item in items if item.prompt_wav is not None},
        key=lambda path: str(path),
    )
    device = torch.device(args.device)

    metrics: dict[str, Any] = {
        "model_path": args.model_path,
        "codec_path": args.codec_path,
        "prompt_wav": str(prompt_wav),
        "prompt_wavs": [str(path) for path in unique_prompt_wavs],
        "prompt_wav_count": len(unique_prompt_wavs),
        "out_dir": str(out_dir),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "attn_implementation": args.attn_implementation,
        "batch_size": args.batch_size,
        "item_count": len(items),
        "packing_mode": args.packing_mode,
        "prefill_text_len": args.prefill_text_len,
        "max_audio_steps": args.max_audio_steps,
        "max_length": args.max_length,
        "sample_rate": args.sample_rate,
        "codec_decode_batch_size": args.codec_decode_batch_size,
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
        "allow_tf32": args.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }

    print(f"[INFO] device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    print(f"[INFO] prompt_wav={prompt_wav}")
    print(f"[INFO] out_dir={out_dir}")
    print(
        f"[INFO] items={len(items)} batch_size={args.batch_size} "
        f"packing_mode={args.packing_mode} max_audio_steps={args.max_audio_steps}"
    )

    tokenizer, processor, _model, codec, inferencer = load_stack(args, device, metrics)
    prefill_text_len = int(args.prefill_text_len or processor.delay_tokens_len)
    metrics["prefill_text_len"] = prefill_text_len

    prefix_cache: dict[Path, np.ndarray] = {}
    prompt_records = []
    for prompt_index, item_prompt_wav in enumerate(unique_prompt_wavs):
        prefix_cache[item_prompt_wav], prompt_record = encode_prompt_prefix(
            args,
            processor,
            tokenizer,
            codec,
            item_prompt_wav,
            device,
            metrics,
        )
        prompt_record["prompt_index"] = prompt_index
        prompt_records.append(prompt_record)
    metrics["prompts"] = prompt_records
    if prompt_records:
        first_prompt = prompt_records[0]
        metrics["prompt_audio_load_s"] = first_prompt.get("prompt_audio_load_s")
        metrics["prompt_audio_duration_s"] = first_prompt.get("prompt_audio_duration_s")
        metrics["prompt_encode_s"] = first_prompt.get("prompt_encode_s")
        metrics["prompt_token_shape"] = first_prompt.get("prompt_token_shape")
        metrics["prefix_build_s"] = first_prompt.get("prefix_build_s")
        metrics["prefix_shape"] = first_prompt.get("prefix_shape")
    metrics["total_prompt_encode_s"] = sum(record.get("prompt_encode_s", 0.0) for record in prompt_records)

    batch_records = []
    for batch_index, (batch_prompt_wav, batch_items) in enumerate(batched_by_prompt(items, args.batch_size)):
        print(
            f"[INFO] batch={batch_index} size={len(batch_items)} "
            f"prompt_wav={batch_prompt_wav} ids={[item.item_id for item in batch_items]}"
        )
        batch_record = generate_microbatch(
            args,
            tokenizer,
            prefill_text_len,
            inferencer,
            codec,
            prefix_cache[batch_prompt_wav],
            batch_items,
            out_dir,
            batch_index,
            metrics,
        )
        batch_record["prompt_wav"] = str(batch_prompt_wav)
        batch_records.append(batch_record)
        print(
            "[BATCH] "
            f"{batch_index}: audio_s={batch_record['audio_duration_s']:.3f} "
            f"wall_s={batch_record['batch_total_s']:.3f} "
            f"audio/sec={batch_record['audio_seconds_per_wall_second']:.3f} "
            f"gen_audio/sec={batch_record['audio_seconds_per_generate_second']:.3f}"
        )

    metrics["batches"] = batch_records
    metrics["process_total_s"] = time.perf_counter() - process_started
    total_audio_s = sum(batch["audio_duration_s"] for batch in batch_records)
    total_batch_wall_s = sum(batch["batch_total_s"] for batch in batch_records)
    total_generate_s = sum(batch["generate_s"] for batch in batch_records)
    total_prefill_s = sum(batch["prefill_s"] for batch in batch_records)
    total_prefill_text_tokens = sum(batch.get("prefill_text_tokens", 0) for batch in batch_records)
    total_text_tokens_stepped = sum(batch.get("text_tokens_stepped", 0) for batch in batch_records)
    total_text_step_calls = sum(batch.get("text_step_calls", 0) for batch in batch_records)
    total_drain_step_calls = sum(batch.get("drain_step_calls", 0) for batch in batch_records)
    metrics["total_audio_duration_s"] = total_audio_s
    metrics["total_batch_wall_s"] = total_batch_wall_s
    metrics["total_prefill_s"] = total_prefill_s
    metrics["total_generate_s"] = total_generate_s
    metrics["total_prefill_text_tokens"] = total_prefill_text_tokens
    metrics["total_text_tokens_stepped"] = total_text_tokens_stepped
    metrics["total_text_step_calls"] = total_text_step_calls
    metrics["total_drain_step_calls"] = total_drain_step_calls
    metrics["audio_seconds_per_batch_wall_second"] = (
        total_audio_s / total_batch_wall_s if total_batch_wall_s else None
    )
    metrics["audio_seconds_per_generate_second"] = total_audio_s / total_generate_s if total_generate_s else None
    metrics["audio_seconds_per_process_second"] = total_audio_s / metrics["process_total_s"] if metrics["process_total_s"] else None
    cuda_peaks = [batch.get("cuda_peak_allocated_mib") for batch in batch_records]
    cuda_reserved_peaks = [batch.get("cuda_peak_reserved_mib") for batch in batch_records]
    metrics["max_cuda_peak_allocated_mib"] = max((x for x in cuda_peaks if x is not None), default=None)
    metrics["max_cuda_peak_reserved_mib"] = max((x for x in cuda_reserved_peaks if x is not None), default=None)
    for stage in ("prefill", "generate", "codec_decode"):
        alloc_key = f"{stage}_cuda_peak_allocated_mib"
        reserved_key = f"{stage}_cuda_peak_reserved_mib"
        metrics[f"max_{alloc_key}"] = max(
            (batch.get(alloc_key) for batch in batch_records if batch.get(alloc_key) is not None),
            default=None,
        )
        metrics[f"max_{reserved_key}"] = max(
            (batch.get(reserved_key) for batch in batch_records if batch.get(reserved_key) is not None),
            default=None,
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if args.benchmark:
        bench_path = benchmark_path(out_dir, args.benchmark_json)
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        if bench_path != manifest_path:
            bench_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print("\n[BENCHMARK]")
    for key in (
        "setup_total_s",
        "prompt_encode_s",
        "packing_mode",
        "prefill_text_len",
        "total_prefill_s",
        "total_generate_s",
        "total_prefill_text_tokens",
        "total_text_tokens_stepped",
        "total_text_step_calls",
        "total_drain_step_calls",
        "codec_batch_decode_s",
        "codec_decode_batch_size",
        "codec_decode_effective_batch_size",
        "total_batch_wall_s",
        "total_audio_duration_s",
        "audio_seconds_per_batch_wall_second",
        "audio_seconds_per_generate_second",
        "audio_seconds_per_process_second",
        "max_prefill_cuda_peak_allocated_mib",
        "max_generate_cuda_peak_allocated_mib",
        "max_codec_decode_cuda_peak_allocated_mib",
        "max_cuda_peak_allocated_mib",
        "max_cuda_peak_reserved_mib",
    ):
        print(f"{key}: {metrics.get(key)}")
    print(f"manifest_json: {manifest_path}")
    print(f"[OK] Wrote {len(items)} WAVs under {out_dir}")


if __name__ == "__main__":
    main()
