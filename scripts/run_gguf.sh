#!/usr/bin/env bash
# Build -> measure -> delete one GGUF variant.
#   scripts/run_gguf.sh bf16          baseline (run first: it saves the BF16 logits for KLD)
#   scripts/run_gguf.sh Q4_K_M        plain llama-quantize
#   scripts/run_gguf.sh Q4_K_M-imat   llama-quantize with Unsloth's imatrix
#   scripts/run_gguf.sh UD-Q4_K_XL    download Unsloth's Dynamic quant
# Env: LIMIT=questions per task (default 100), KEEP=1 to keep the weights afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1
OUT=results/gguf-$NAME
BASE=models/gguf/Qwen3-1.7B-BF16.gguf
MODEL=models/gguf/$NAME.gguf
mkdir -p "$OUT"

# 1. Build (quantization time + peak memory go to quant.log)
case $NAME in
  bf16)   MODEL=$BASE ;;
  UD-*)   .venv/bin/hf download unsloth/Qwen3-1.7B-GGUF "Qwen3-1.7B-$NAME.gguf" --local-dir models/gguf >/dev/null
          mv "models/gguf/Qwen3-1.7B-$NAME.gguf" "$MODEL" ;;
  *-imat) /usr/bin/time -l llama-quantize --imatrix models/gguf/imatrix_unsloth.dat "$BASE" "$MODEL" "${NAME%-imat}" > "$OUT/quant.log" 2>&1 ;;
  *)      /usr/bin/time -l llama-quantize "$BASE" "$MODEL" "$NAME" > "$OUT/quant.log" 2>&1 ;;
esac
stat -f %z "$MODEL" > "$OUT/size_bytes"

# 2. Speed: prompt processing (pp512) and generation (tg128), 3 repetitions
llama-bench -m "$MODEL" -p 512 -n 128 -r 3 -o json > "$OUT/bench.json" 2>/dev/null

# 3. Drift vs BF16 on wikitext-2 (20 x 512-token chunks)
if [ "$NAME" = bf16 ]; then
  llama-perplexity -m "$MODEL" -f data/wiki.test.raw -c 512 --chunks 20 --kl-divergence-base results/gguf-base.kld > "$OUT/kld.txt" 2>&1
else
  llama-perplexity -m "$MODEL" --kl-divergence-base results/gguf-base.kld --kl-divergence > "$OUT/kld.txt" 2>&1
fi

# 4. Tasks + latency, with the server wrapped in /usr/bin/time for peak memory footprint
/usr/bin/time -l llama-server -m "$MODEL" -c 4096 --port 8080 --jinja > "$OUT/server.log" 2>&1 &
TIME_PID=$!
until curl -sf localhost:8080/health >/dev/null; do sleep 1; done
.venv/bin/python scripts/eval.py --name "gguf-$NAME" --limit "${LIMIT:-100}" | tail -12
kill -INT "$(pgrep -P $TIME_PID)"; wait $TIME_PID || true

# 5. Rolling: free the disk
[ "$NAME" != bf16 ] && [ -z "${KEEP:-}" ] && rm -f "$MODEL"
echo "done: $OUT"
