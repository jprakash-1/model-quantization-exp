# Quantization Lab: plan (Kaggle edition)

Goal: learn what quantization costs and buys by measuring **one model** against its 16-bit original. For every format, measure **memory, speed, latency and quality**.
This is a lab notebook, not a product: one config file, a few short modules, one master notebook.

Everything runs on **Kaggle** (free NVIDIA T4). Nothing heavy runs locally: no weights are downloaded and no inference engines are installed on this Mac.

---

## 1. Why Kaggle, and what that changes

| Kaggle fact | Consequence |
|---|---|
| NVIDIA **T4 16 GB** (compute capability 7.5), 2 per session | All the CUDA formats work: bitsandbytes, HQQ, GPTQ, llama.cpp CUDA. We use only GPU 0, so memory numbers stay clean. |
| T4 has **no BF16 or FP8 hardware** | The 16-bit baseline is **FP16** (the code picks BF16 automatically on an A100/L4). FP8 is in the list *on purpose*: watching it fail is the lesson "your GPU decides the format". |
| ~70 GB disk, 20 GB of it persisted in `/kaggle/working` | Big files go to `/tmp/qlab`, results to the repo's `results/` (persisted). GGUF quants are deleted after measuring (rolling). |
| 12 h max per session, ~30 GPU-h per week | Every variant saves its own `metrics.json`. Re-running skips finished ones, so a timeout loses nothing. |

## 2. The model

**`Qwen/Qwen3-1.7B`** with thinking mode off, greedy decoding.
- FP16 is 3.4 GB, so even the 2-bit variants load fast and a full run fits in one session.
- Small models are *more* sensitive to quantization, which makes the effects easy to see.
- Every format has a ready source: the original weights, `unsloth/Qwen3-1.7B-GGUF` (BF16 + imatrix + `UD-*`), `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`, `Qwen/Qwen3-1.7B-GPTQ-Int8`, and `Qwen/Qwen3-1.7B-FP8`.
- `config.set_model("Qwen/Qwen3-4B")` repeats everything on a bigger model.

## 3. Variants (two tracks, each against its own baseline)

**Track 1: Hugging Face Transformers** (`qlab/hf_backend.py`)

| Variant | Method | Question it answers |
|---|---|---|
| `fp16` | baseline | |
| `bnb-int8` | LLM.int8() with 16-bit outliers | Is 8-bit free? Is it fast? |
| `bnb-nf4` / `bnb-fp4` | 4-bit NormalFloat vs 4-bit float | Does the code book matter? |
| `bnb-nf4-dq` | NF4 + double-quantized scales | What do the scales cost? |
| `unsloth-dyn4` | NF4, sensitive layers kept 16-bit | Uniform vs mixed precision |
| `hqq4` / `hqq3` / `hqq2` | calibration-free, group 64 | Where is the cliff? |
| `fp8` | Qwen's FP8 checkpoint | expected to fail on a T4 |
| `torchao-int8`, `gptq-int8` | optional, need an extra install | |

**Track 2: llama.cpp GGUF** (`qlab/gguf_backend.py`), built from source with CUDA on Kaggle

| Variant | Question it answers |
|---|---|
| `f16` | baseline |
| `Q8_0`, `Q6_K`, `Q5_K_M` | the near-lossless zone |
| `Q4_K_M` → `Q4_K_M-imat` → `UD-Q4_K_XL` | what calibration, then per-layer mixing, buy at 4-bit |
| `IQ4_XS-imat` | non-uniform code book |
| `Q3_K_M-imat` | the 3-bit cliff |
| `Q2_K` → `Q2_K-imat` → `UD-Q2_K_XL` | can calibration and mixing rescue 2-bit? |

## 4. Metrics (open datasets only)

| Group | Metric | How |
|---|---|---|
| Memory | weights size (GB), bits per weight, **peak GPU memory** | `get_memory_footprint()` / GGUF file size. Peak is NVML-sampled every 50 ms, identically for in-process (HF) and `llama-server`. |
| Speed | **prefill tok/s**, **decode tok/s**, **TTFT** (512-token prompt) | single stream, median of 3. HF: timed `forward` + `generate`. GGUF: `llama-bench`. |
| Cost to make | quantization time, load time | wall clock |
| Drift | **mean / p99 KL divergence** vs baseline, **top-1 agreement**, perplexity | `Salesforce/wikitext` wikitext-2 test, 20 × 512 tokens. HF: our own code against saved baseline log-probs. GGUF: `llama-perplexity --kl-divergence`. |
| Quality | **GSM8K** strict (`#### N`) and flexible (last number) | `openai/gsm8k`, first 250 test questions |
| | **ARC-Challenge** | `allenai/ai2_arc`, first 250 test questions |
| | **answers lost / gained vs baseline** | paired per question, more sensitive than accuracy at n=250 (SE ≈ ±2.5 pts) |
| | hit max tokens | rambling or repetition loops, a typical low-bit failure |

Rules that keep it fair: same prompts, same chat template, thinking off, greedy decoding, same questions in the same order. Never compare KLD or speed **across** tracks, only within one.

## 5. Code layout

```
model-quantization-exp/
├── kaggle_master.ipynb     # the only thing you run
├── requirements-kaggle.txt
├── PLAN.md
├── qlab/
│   ├── config.py           # all knobs (model, N questions, chunks, batch)
│   ├── data.py             # download wikitext-2, GSM8K, ARC → data/
│   ├── tasks.py            # prompts + scoring; works with any generate() function
│   ├── gpu.py              # PeakGpuMemory (NVML sampler)
│   ├── hf_backend.py       # Transformers variants, batched generate, bench, KLD
│   ├── gguf_backend.py     # build llama.cpp, quantize, bench, KLD, llama-server
│   ├── run.py              # run one variant end to end; resume; record failures
│   └── report.py           # summary.csv / .md / .png
└── results/<track>-<variant>/   metrics.json, eval.json (every reply), logs
```

## 6. How to run

1. Push this repo to GitHub (public, or private with a token).
2. On Kaggle: **New Notebook → File → Import Notebook →** `kaggle_master.ipynb`, or just paste the cells.
3. Sidebar: **Accelerator = GPU T4 ×2**, **Internet = On**.
4. Edit `REPO_URL` in cell 1.
5. First time, set `config.N_QUESTIONS = 100` and run the cells interactively to catch problems early.
6. Then set it to 250 and **Save Version → Save & Run All (Commit)** to run unattended.
7. Download `results/` from the notebook output, or push it back with the last cell.

Rough time on a T4 at n=250: HF track ~2–3 h (HQQ and bnb-int8 are the slow ones), llama.cpp build ~15 min, GGUF track ~1.5 h.

## 7. Questions the results should answer

These are in the last notebook cell too. Write the answers into `FINDINGS.md`.
1. Is 8-bit indistinguishable from the baseline? Is `bnb-int8` actually *slower* than FP16?
2. At 4-bit, which matters most: the code book, the scales, calibration, or per-layer mixing?
3. Does perplexity move under 1% while KLD and GSM8K clearly move?
4. Does GSM8K drop before ARC?
5. Do some quants break the answer *format* (strict ≪ flexible) before the math?
6. Where is the cliff for 1.7B: 3-bit or 2-bit? Do imatrix or UD move it?
7. Which formats are smaller but *not* faster (dequantization overhead)?
8. Which variants use more GPU memory than their bits suggest, and why?
9. What would you deploy on a 16 GB GPU?

## 8. Known risks

- **Library drift on Kaggle.** The notebook installs the latest `transformers` / `bitsandbytes` / `hqq`. If one breaks, that variant fails with `error.txt` and the rest continue. Pin a version in `requirements-kaggle.txt` to fix it.
- **`gptqmodel` compiles CUDA code** (10+ min) and `torchao` must match the torch version, so both are optional cells.
- **Batched HF generation** pads on the left. That can change a few greedy tokens compared with batch size 1. It's applied the same way to every HF variant, so comparisons within the track stay fair.
- **The GGUF baseline is F16 and the HF baseline is FP16.** Each track's KLD is relative to its own baseline.
- **Kaggle's internet toggle needs a phone-verified account.**
