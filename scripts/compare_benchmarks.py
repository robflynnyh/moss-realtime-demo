#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


FIELDS = (
    "decode_chunk_frames",
    "first_audio_chunk_s",
    "chunk_interval_p50_s",
    "chunk_interval_p95_s",
    "audio_seconds_per_wall_second",
    "audio_seconds_per_wall_second_after_first_chunk",
    "model_stream_calls_pct_stream",
    "codec_decode_s_pct_stream",
    "rtf_stream",
    "audio_duration_s",
    "audio_chunk_count",
)


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: scripts/compare_benchmarks.py outputs/*.benchmark.json")

    rows = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        data = json.loads(path.read_text())
        data["file"] = path.name
        rows.append(data)

    headers = ("file",) + FIELDS
    table = [[fmt(row.get(header)) for header in headers] for row in rows]
    widths = [
        max(len(header), *(len(row[idx]) for row in table))
        for idx, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in table:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


if __name__ == "__main__":
    main()
