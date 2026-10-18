# Quantization Experiment Plan — one model, BF16 vs every quant

> **Update (2026-10-18), as run:** with only ~21 GB of free disk, the model is **`Qwen/Qwen3-1.7B`** (BF16 ≈ 3.4–4 GB, thinking mode off),
> run **rolling**: `scripts/run_gguf.sh <variant>` and `scripts/run_mlx.sh <variant>` each build → measure → delete one variant.
> The BF16 GGUF comes from `unsloth/Qwen3-1.7B-GGUF` (no conversion step), and the imatrix is `imatrix_unsloth.dat`.
> MLX loads the original HF safetensors directly as its BF16 baseline. At the end, all weights and caches under `models/` are deleted. Only scripts and `results/` are kept in git.
> Wherever the text below says Qwen3-4B, read Qwen3-1.7B. Speeds are about 2.3× higher than the 4B estimates.

Goal: learn what quantization costs and buys by measuring one model on your own Mac.
For each format, measure **memory, speed, latency and quality** against BF16.
This is a lab notebook, not a product, so keep every script short and every step easy to repeat.

---

## 0. Constraints that shape the plan

| Fact | Consequence |
|---|---|
| Apple M3, **16 GB unified memory**, no NVIDIA GPU | CUDA-only tools don't run here: bitsandbytes, GPTQModel/AutoAWQ kernels, vLLM, FP8 matmuls. We use the two engines that are fast on Apple Silicon: **llama.cpp (GGUF)** and **MLX**. |
| BF16 needs 2 bytes/param, and macOS + apps need ~4–5 GB | The model must be ≤ ~4B params so that BF16 fits with room for the KV cache. |
| M3 memory bandwidth ≈ 100 GB/s | Decode speed is roughly `bandwidth ÷ model size`, so speed should scale almost linearly with quant size. You'll test that prediction. |
| Unsloth's *library* targets NVIDIA GPUs | We use Unsloth's **published Dynamic GGUFs** (`UD-*`) and its imatrix file, not its training code. |

## 1. The model

**`Qwen/Qwen3-4B-Instruct-2507`** (4.0B params, non-thinking instruct, 152K vocab)

Why this one:
- BF16 is ~8 GB, which fits in 16 GB with room to spare.
- Every format we need already exists or is one command away:
  - `Qwen/Qwen3-4B-Instruct-2507`: original BF16 safetensors
  - `unsloth/Qwen3-4B-Instruct-2507-GGUF`: full ladder Q8_0 to Q2_K, `UD-Q*_K_XL` Dynamic quants, IQ quants, and `imatrix_unsloth.gguf_file`
  - `mlx-community/Qwen3-4B-Instruct-2507-bf16`: MLX baseline
- At 4B, quantization damage is *visible*. Small models are more sensitive than the 27B in the Ledger, which makes the lesson clearer.
- It's non-thinking, so answers are short and GSM8K runs stay fast.

> Note: Unsloth's repo ships `F16`, not BF16. We make our own BF16 GGUF from the original weights, so the baseline is true BF16.

---

## 2. What we compare (the variant matrix)

About 20 variants in two tracks. Each track is compared to **its own BF16**, because KL divergence only makes sense inside one engine.

### Track A: llama.cpp / GGUF

| # | Variant | Source | What it teaches |
|---|---|---|---|
| A0 | **BF16** | convert yourself | baseline |
| A1 | Q8_0 | quantize yourself | "8-bit is free" |
| A2 | Q6_K | quantize yourself | |
| A3 | Q5_K_M | quantize yourself | |
| A4 | Q4_K_M (no imatrix) | quantize yourself | plain k-quant |
| A5 | Q4_K_M (imatrix) | quantize yourself with Unsloth's imatrix | what calibration buys at 4-bit |
| A6 | IQ4_XS (imatrix) | quantize yourself | non-uniform 4-bit codebook |
| A7 | Q3_K_M (imatrix) | quantize yourself | the 3-bit cliff |
| A8 | Q2_K (no imatrix) | quantize yourself | naive 2-bit collapse |
| A9 | Q2_K (imatrix) | quantize yourself | calibration at 2-bit |
| A10 | UD-Q4_K_XL | download Unsloth | per-layer mixed precision vs A5 |
| A11 | UD-Q2_K_XL | download Unsloth | mixed precision vs A9 |

### Track B: MLX (`mlx-lm` 0.32)

| # | Variant | Command | What it teaches |
|---|---|---|---|
| B0 | **BF16** | download | baseline |
| B1 | 8-bit affine, g64 | `mlx_lm.convert -q --q-bits 8` | |
| B2 | 6-bit affine | `--q-bits 6` | |
| B3 | 4-bit affine, g64 | `--q-bits 4` | plain RTN with groups |
| B4 | 4-bit affine, **g32** | `--q-bits 4 --q-group-size 32` | effect of group size |
| B5 | 3-bit affine | `--q-bits 3` | the 3-bit cliff |
| B6 | mixed 3/6 | `--quant-predicate mixed_3_6` | mixed precision, MLX style |
| B7 | MXFP4 | `--q-mode mxfp4` | 4-bit *float* format vs 4-bit int |
| B8 | 4-bit **AWQ** | `mlx_lm.awq --bits 4` | activation-aware calibration |
| B9 | 4-bit **GPTQ** | `mlx_lm.gptq --bits 4` | Hessian-based error correction |
| B10 | 4-bit **DWQ** *(optional)* | `mlx_lm.dwq` | distillation-tuned quant. Slow and memory-hungry; skip if it OOMs |
| B11 | dynamic ~4.5 bpw *(optional)* | `mlx_lm.dynamic_quant --target-bpw 4.5` | sensitivity-chosen bits per layer |

**Not in scope on this Mac:** bitsandbytes NF4/INT8, GPTQ/AWQ CUDA kernels, FP8 in vLLM. To try them later, run the same eval script on a free Colab T4 (see §9).

---

## 3. What we measure

### Efficiency
| Metric | How | Tool |
|---|---|---|
| File size on disk (GB) and **bits per weight** | `size × 8 ÷ 4.02e9` | `du` |
| **Peak memory footprint** (GB) | `/usr/bin/time -l` → "peak memory footprint". This includes Metal/unified memory, unlike RSS. | macOS |
| Weight buffer + KV cache size | printed at load | llama.cpp log / `mlx_lm.generate` "Peak memory" |
| **Prefill speed** (tok/s, 512-token prompt) | `pp512` | `llama-bench`, `mlx_lm.benchmark` |
| **Decode speed** (tok/s, 128 tokens) | `tg128` | same |
| **TTFT** and end-to-end latency (ms) | streamed request to the local server, timed in Python | `eval.py` |
| Load time (s) | wall time to first ready | server log |

### Quality (open datasets)
| Metric | Dataset | Why it's here |
|---|---|---|
| **KL divergence vs BF16** (mean, p99) and **top-1 token agreement** | `Salesforce/wikitext` wikitext-2-raw-v1 test, 20 × 512-token chunks | The most sensitive signal. It catches drift that accuracy misses (Ledger finding: "perplexity hides most of the damage"). |
| Perplexity | same | Included to show *how little* it moves. |
| **GSM8K** strict and flexible accuracy | `openai/gsm8k` main/test, first 250 questions, 0-shot chat, greedy | Free-form generation with a strict answer format, which is where 4-bit hurts first. |
| **ARC-Challenge** accuracy | `allenai/ai2_arc` ARC-Challenge/test, first 250 questions | Multiple choice. Expect it to barely move, which is the contrast with GSM8K. |
| **Answer flips vs BF16** | same questions | How many answers changed right→wrong and wrong→right. A paired comparison is far more informative than raw accuracy at n=250. |

> **Noise check:** at n=250 and ~85% accuracy, the standard error is about ±2.3 pts, so differences under ~5 pts are noise. That's why KLD and paired flips matter.

---

## 4. Folder layout (keep it flat)

```
model-quantization-exp/
├── PLAN.md
├── Quantization Ledger.html
├── models/            # all weights (large, never commit)
│   ├── hf-bf16/       # original safetensors
│   ├── gguf/          # A0–A11
│   └── mlx/           # B0–B11
├── data/              # wiki.test.raw, gsm8k.jsonl, arc.jsonl
├── scripts/
│   ├── get_data.py    # download the 3 datasets → data/
│   ├── eval.py        # GSM8K + ARC + TTFT against any OpenAI-compatible server
│   ├── kld_mlx.py     # KLD / top-1 agreement for MLX variants
│   └── report.py      # merge results/*.json → one table + 4 plots
└── results/           # one JSON per variant, plus bench logs
```

There's one eval script for both engines, because `llama-server` and `mlx_lm.server` both expose `/v1/chat/completions`. Only the server command changes.

---

## 5. Setup (≈30 min)

```bash
# Python env
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install mlx-lm datasets huggingface_hub openai pandas matplotlib

# llama.cpp (Metal is on by default on macOS)
brew install llama.cpp            # gives llama-server, llama-bench, llama-perplexity, llama-quantize, llama-imatrix
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp   # only for convert_hf_to_gguf.py
uv pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt

# Weights
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/hf-bf16
hf download mlx-community/Qwen3-4B-Instruct-2507-bf16 --local-dir models/mlx/bf16
hf download unsloth/Qwen3-4B-Instruct-2507-GGUF \
  --include "*UD-Q4_K_XL*" "*UD-Q2_K_XL*" "imatrix_unsloth.gguf_file" --local-dir models/gguf
```

Before every benchmark, quit browsers, IDEs and Docker, and keep the Mac plugged in. Memory pressure and low-power mode distort the speed numbers.

---

## 6. Phases

### Phase 1: Predict first (15 min, no code)
For each variant, write down a guess for size, decode tok/s (`100 GB/s ÷ size`) and whether GSM8K will drop. You'll compare these against the measurements at the end. This is the fastest way to build intuition.

### Phase 2: Build the variants (≈1 h, mostly waiting)

**GGUF**
```bash
python ~/llama.cpp/convert_hf_to_gguf.py models/hf-bf16 --outtype bf16 --outfile models/gguf/bf16.gguf

IM=models/gguf/imatrix_unsloth.gguf_file
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q2_K; do
  llama-quantize models/gguf/bf16.gguf models/gguf/$q.gguf $q
done
for q in Q4_K_M IQ4_XS Q3_K_M Q2_K; do
  llama-quantize --imatrix $IM models/gguf/bf16.gguf models/gguf/$q-imat.gguf $q
done
```
*(Optional learning step: build your own imatrix with `llama-imatrix -m bf16.gguf -f data/calib.txt`, then compare it with Unsloth's.)*

**MLX**
```bash
B=models/hf-bf16
mlx_lm.convert --hf-path $B --mlx-path models/mlx/q8      -q --q-bits 8
mlx_lm.convert --hf-path $B --mlx-path models/mlx/q6      -q --q-bits 6
mlx_lm.convert --hf-path $B --mlx-path models/mlx/q4      -q --q-bits 4
mlx_lm.convert --hf-path $B --mlx-path models/mlx/q4-g32  -q --q-bits 4 --q-group-size 32
mlx_lm.convert --hf-path $B --mlx-path models/mlx/q3      -q --q-bits 3
mlx_lm.convert --hf-path $B --mlx-path models/mlx/mix36   -q --quant-predicate mixed_3_6
mlx_lm.convert --hf-path $B --mlx-path models/mlx/mxfp4   -q --q-mode mxfp4
mlx_lm.awq  --model $B --mlx-path models/mlx/awq4  --bits 4
mlx_lm.gptq --model $B --mlx-path models/mlx/gptq4 --bits 4
# optional: mlx_lm.dwq --model $B --quantized-model models/mlx/q4 --mlx-path models/mlx/dwq4 --grad-checkpoint
```
For each build, note **quantization time and peak memory** (`/usr/bin/time -l`). The cost of producing a quant is a metric too (Ledger: "Cost of producing a 4-bit model").

### Phase 3: Data (10 min)
`get_data.py` writes three files and nothing else:
- `data/wiki.test.raw`: the wikitext-2 test split joined into one text file
- `data/gsm8k.jsonl`: `{question, answer_number}` × 250
- `data/arc.jsonl`: `{question, choices, label}` × 250

### Phase 4: Speed and memory (≈2 min per variant)
```bash
# GGUF: prefill/decode, 3 repetitions
llama-bench -m models/gguf/Q4_K_M.gguf -p 512 -n 128 -r 3 -o json > results/bench_A4.json

# MLX
mlx_lm.benchmark --model models/mlx/q4 -p 512 -g 128 -n 3 > results/bench_B3.txt

# Peak memory: one realistic generation under /usr/bin/time
/usr/bin/time -l llama-cli -m models/gguf/Q4_K_M.gguf -p "Explain KV cache." -n 256 -no-cnv 2>&1 | grep "peak memory footprint"
```

### Phase 5: Distribution drift, KLD (≈5 min per variant)
**GGUF.** llama.cpp has this built in:
```bash
# once: save BF16 logits (~3 GB file because of the 152K vocab; that's why we use only 20 chunks)
llama-perplexity -m models/gguf/bf16.gguf -f data/wiki.test.raw -c 512 --chunks 20 \
  --kl-divergence-base results/bf16.kld
# each quant
llama-perplexity -m models/gguf/Q4_K_M.gguf --kl-divergence-base results/bf16.kld --kl-divergence \
  > results/kld_A4.txt   # prints PPL, mean KLD, p99 KLD, "Same top p"
```
**MLX.** `kld_mlx.py` (~40 lines):
1. Tokenize the wikitext text once and cut 20 × 512-token chunks.
2. `--dump`: run BF16 and save the log-softmax for each chunk to disk as float16. This avoids holding two models in 16 GB.
3. `--compare <model>`: run the quant, then compute `KL(p_bf16‖p_q)` per token. Report mean, p99, top-1 agreement and PPL.

### Phase 6: Task accuracy and latency (the long one)
Start a server, run `eval.py`, stop the server:
```bash
llama-server -m models/gguf/Q4_K_M.gguf -c 4096 --port 8080        # GGUF
mlx_lm.server --model models/mlx/q4 --port 8080                    # MLX
python scripts/eval.py --name A4 --port 8080
```
`eval.py` does exactly this:
- For each GSM8K question, send a chat request asking it to "end with `#### <number>`" at temperature 0 with max_tokens 512, **streamed**. Record TTFT, total time and output tokens.
- **strict** = the `#### N` line parses and equals the gold answer. **flexible** = the last number in the reply equals the gold answer.
- For each ARC question, ask for one letter with max_tokens 5. Record correct or not.
- Save `results/eval_<name>.json` with per-question correctness. The per-question data is what makes the flip analysis possible.

Time budget: BF16 decodes at ~10–12 tok/s on an M3, so a GSM8K run takes ~1.5 h. Q4 is ~3× faster. Run the slow ones overnight with `caffeinate -i`.
**Shortcut:** screen all variants on 100 questions first, then run the full 250 on the interesting ones.

### Phase 7: Report
`report.py` merges everything into one table (one row per variant):

```
variant | bpw | disk GB | peak mem GB | pp512 t/s | tg128 t/s | TTFT ms | PPL | mean KLD | top-1 % | GSM8K strict | GSM8K flex | ARC | flips vs BF16
```
It also makes four plots:
1. **Decode tok/s vs file size**, with the `100 GB/s ÷ size` line overlaid. Is decode really bandwidth-bound?
2. **Mean KLD vs bits per weight** (log y), one line per track. Where is the knee?
3. **GSM8K and ARC vs bits per weight.** Which benchmark breaks first?
4. **Pareto: peak memory vs GSM8K.** Which variants aren't beaten on both axes?

---

## 7. Questions the results should answer

1. Is Q8_0 / MLX-8bit indistinguishable from BF16? (KLD < ~0.005, zero net flips?)
2. At 4-bit, which matters more: the **method** (RTN vs AWQ vs GPTQ vs imatrix vs UD) or the **group size**?
3. Does PPL move less than 1% while KLD and GSM8K clearly move? (This reproduces the Ledger's "perplexity hides damage" finding.)
4. Does GSM8K drop before ARC does? (This reproduces "the benchmark decides what you see".)
5. Is GSM8K *strict minus flexible* larger for some quants, meaning they break the format before the math?
6. Where is the cliff for a 4B model, at 3-bit or 2-bit? Does imatrix or UD move it?
7. Does decode speed follow `bandwidth ÷ size`, and does prefill (compute-bound) also speed up, or not?
8. At the same bits, which is faster on M3: llama.cpp or MLX?
9. Your pick: which single variant would you run day to day on this Mac, and why?

Write the answers in a short `FINDINGS.md`, with your Phase 1 predictions next to them.

---

## 8. Pitfalls

- **Same prompts, same chat template, temperature 0** for every variant, or the comparison is meaningless. Keep the system prompt fixed in `eval.py`.
- **Never compare KLD across tracks.** GGUF-BF16 and MLX-BF16 differ in numerics. Compare tracks only on task accuracy and speed.
- **Never compare PPL across models or tokenizers.** Within this one model it's fine.
- **The first run is slow** (Metal shader compile, cold disk cache). Benchmarks use 3 repetitions; discard a warm-up.
- **Thermals:** a fanless MacBook Air M3 throttles on long runs. Note which machine you're on, and run benches before the long evals, not after.
- **KV cache memory** is outside the weight size. Peak memory is measured with a fixed `-c 4096`, so all variants are on equal footing.
- **Disk:** about 60 GB for all variants plus the KLD file. Delete variants once their eval JSON exists.

## 9. Optional extension: the CUDA formats (free Colab T4)
On Colab, load `Qwen/Qwen3-4B-Instruct-2507` in Transformers with `BitsAndBytesConfig(load_in_8bit)` and `load_in_4bit, nf4` (and optionally a GPTQModel/AWQ checkpoint). Serve it via vLLM, or just call `generate`, and run the same `eval.py` question sets. This adds bitsandbytes and GPTQ/AWQ-CUDA rows to the table. Compare only their accuracy and flips with the Mac rows, because their speed is on different hardware.

---

## 10. Order of work (checklist)
- [ ] Setup, downloads, BF16 GGUF conversion
- [ ] Phase 1 predictions written down
- [ ] Build A1–A9, B1–B9
- [ ] `get_data.py`
- [ ] Bench + peak memory for all variants (fast; do first)
- [ ] KLD for all variants
- [ ] 100-question screen on GSM8K/ARC for all variants
- [ ] Full 250 on BF16 + the 4–6 most interesting variants
- [ ] `report.py` table + 4 plots
- [ ] `FINDINGS.md`
