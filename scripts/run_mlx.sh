#!/usr/bin/env bash
# Build -> measure -> delete one MLX variant.
#   scripts/run_mlx.sh bf16 | q8 | q6 | q4 | q4-g32 | q3 | mix36 | mxfp4 | awq4 | gptq4 | dwq4
# Env: LIMIT=questions per task (default 100), KEEP=1 to keep the weights afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1
OUT=results/mlx-$NAME
BASE=models/hf-bf16
MODEL=models/mlx/$NAME
BIN=.venv/bin
mkdir -p "$OUT"

# 1. Build (quantization time + peak memory go to quant.log)
build() { /usr/bin/time -l "$@" > "$OUT/quant.log" 2>&1; }
convert() { build $BIN/mlx_lm.convert --hf-path $BASE --mlx-path "$MODEL" -q "$@"; }
rm -rf "$MODEL"
case $NAME in
  bf16)   MODEL=$BASE ;;  # MLX loads the original HF safetensors directly
  q8)     convert --q-bits 8 ;;
  q6)     convert --q-bits 6 ;;
  q4)     convert --q-bits 4 ;;
  q4-g32) convert --q-bits 4 --q-group-size 32 ;;
  q3)     convert --q-bits 3 ;;
  mix36)  convert --quant-predicate mixed_3_6 ;;
  mxfp4)  convert --q-mode mxfp4 ;;
  awq4)   build $BIN/mlx_lm.awq --model $BASE --mlx-path "$MODEL" --bits 4 ;;
  gptq4)  build $BIN/mlx_lm.gptq --model $BASE --mlx-path "$MODEL" --bits 4 ;;
  dwq4)   convert --q-bits 4 && mv "$MODEL" "$MODEL-init"
          build $BIN/mlx_lm.dwq --model $BASE --quantized-model "$MODEL-init" --mlx-path "$MODEL" --bits 4 --grad-checkpoint
          rm -rf "$MODEL-init" ;;
  *) echo "unknown variant $NAME"; exit 1 ;;
esac
du -sk "$MODEL" | awk '{print $1 * 1024}' > "$OUT/size_bytes"

# 2. Speed: prompt 512 / generate 128, 3 trials
$BIN/mlx_lm.benchmark --model "$MODEL" -p 512 -g 128 -n 3 > "$OUT/bench.txt" 2>&1

# 3. Drift vs BF16 on wikitext-2
[ "$NAME" != bf16 ] && $BIN/python scripts/kld_mlx.py --quant "$MODEL" --out "$OUT/kld.json" | tail -7

# 4. Tasks + latency, with the server wrapped in /usr/bin/time for peak memory footprint
/usr/bin/time -l $BIN/mlx_lm.server --model "$MODEL" --port 8080 > "$OUT/server.log" 2>&1 &
TIME_PID=$!
until curl -sf localhost:8080/v1/models >/dev/null; do sleep 1; done
$BIN/python scripts/eval.py --name "mlx-$NAME" --limit "${LIMIT:-100}" | tail -12
kill -INT "$(pgrep -P $TIME_PID)"; wait $TIME_PID || true

# 5. Rolling: free the disk
[ "$NAME" != bf16 ] && [ -z "${KEEP:-}" ] && rm -rf "$MODEL"
echo "done: $OUT"
