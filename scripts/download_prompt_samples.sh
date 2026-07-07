#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_DIR="$ROOT_DIR/prompts"
RAW_DIR="$PROMPT_DIR/raw"

mkdir -p "$PROMPT_DIR" "$RAW_DIR"

download_and_trim() {
  local name="$1"
  local url="$2"
  local start_s="$3"
  local duration_s="$4"
  local extension="${5:-ogg}"
  local raw_path="$RAW_DIR/${name}.${extension}"
  local wav_path="$PROMPT_DIR/${name}_12s.wav"

  if [[ ! -s "$raw_path" ]]; then
    curl -fL "$url" -o "$raw_path"
  fi

  ffmpeg -hide_banner -loglevel error -y \
    -ss "$start_s" -t "$duration_s" -i "$raw_path" \
    -ac 1 -ar 24000 \
    -af "loudnorm=I=-20:TP=-2:LRA=11" \
    "$wav_path"

  python3 - "$wav_path" <<'PY'
import sys
import wave

path = sys.argv[1]
with wave.open(path, "rb") as wav:
    print(
        f"{path}: {wav.getnchannels()}ch {wav.getframerate()}Hz "
        f"{wav.getnframes() / wav.getframerate():.2f}s"
    )
PY
}

download_and_trim \
  "jfk_berlin" \
  "https://commons.wikimedia.org/wiki/Special:Redirect/file/Jfk%20berlin%20address%20high.ogg" \
  53 12 ogg

download_and_trim \
  "nixon_resignation" \
  "https://commons.wikimedia.org/wiki/Special:Redirect/file/Nixon%20resignation%20audio%20with%20buzz%20removed.ogg" \
  36 12 ogg

download_and_trim \
  "fdr_fireside" \
  "https://commons.wikimedia.org/wiki/Special:Redirect/file/FDR%20Chat%20Mar%2037.ogg" \
  24 12 ogg

download_and_trim \
  "nabu_joe_en_us" \
  "https://github.com/NabuCasa/voice-datasets/raw/master/en_US/joe/0000000001.mp3" \
  0 12 mp3

download_and_trim \
  "nabu_kathleen_en_us" \
  "https://github.com/NabuCasa/voice-datasets/raw/master/en_US/kathleen/arctic_a0001_1592748574.mp3" \
  0 12 mp3

download_and_trim \
  "nabu_kerstin_de_de" \
  "https://github.com/NabuCasa/voice-datasets/raw/master/de_DE/kerstin/de_rhasspy-0004.mp3" \
  0 12 mp3

download_and_trim \
  "nabu_dave_es_es" \
  "https://github.com/NabuCasa/voice-datasets/raw/master/es_ES/dave/0000000001.mp3" \
  0 12 mp3

cat <<'EOF'

Prompt WAVs are ready in ./prompts.
See prompts/MANIFEST.md for source and license notes.
EOF
