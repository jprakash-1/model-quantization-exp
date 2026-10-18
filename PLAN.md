# Quantization Lab: plan (Kaggle edition)

Goal: learn what quantization costs and buys by measuring **one model** against its 16-bit original. For every format, measure **memory, speed, latency and quality**.
This is a lab notebook, not a product: one config file, a few short modules, one master notebook.

Everything runs on **Kaggle** on the free **2× NVIDIA T4** accelerator. Nothing heavy runs locally: no weights are downloaded and no inference engines are installed on this Mac.

---

## 1. Why Kaggle, and what that changes

| Kaggle fact | Consequence |
|---|---|
| **2 × Tesla T4, 15 GB each** (Turing, compute capability 7.5, ~320 GB/s), driver 580 / CUDA 13.0 | Everything runs: bitsandbytes, HQQ, llama.cpp CUDA (with fast int8 kernels). |
| **Two GPUs** | **One track per GPU, in parallel.** The HF track runs in the notebook kernel on GPU 0. The GGUF track runs as a background job (`python -m qlab.run gguf ...`) on GPU 1. The model is never split across GPUs, so memory numbers stay clean. |
| T4 has **no BF16, no FP8** | The 16-bit baseline is **FP16** (the code picks BF16 automatically on an A100/L4). FP8 is in the list *on purpose*: transformers dequantizes it to 16-bit on a T4, so it measures only the FP8 rounding error, with no memory or speed gain. Lesson: "your GPU decides the format". |
| 4 CPU cores shared by both tracks | llama.cpp is compiled *before* either track starts measuring, and `llama-quantize` is limited to 2 threads. NVML ignores `CUDA_VISIBLE_DEVICES`, so `gpu.py` maps it to the right physical GPU. |
| ~70 GB disk, 20 GB of it persisted in `/kaggle/working` | Big files go to `/tmp/qlab`, results to the repo's `results/` (persisted). GGUF quants are deleted after measuring (rolling). |
| 12 h max per session, ~30 GPU-h per week | Every variant saves its own `metrics.json`. Re-running skips finished ones, so a timeout loses nothing. |

## 2. The model

**`Qwen/Qwen3-4B-Instruct-2507`**: 4.02B params, non-thinking instruct, greedy decoding.
- FP16 is 8 GB. That fits one 15 GB T4 with room for batched generation and the KV cache. An 8B model would be 16 GB in FP16 and wouldn't fit one T4.
- It's big enough to be useful, and small enough that 3-bit and 2-bit damage is clearly visible.
- Every format has a ready source: the original weights, `unsloth/Qwen3-4B-Instruct-2507-GGUF` (F16 + imatrix + `UD-*`), `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`, and `Qwen/Qwen3-4B-Instruct-2507-FP8`.
- Not chosen: **Qwen3.5-4B** (4.66B, multimodal, plus hybrid linear-attention layers, which are less mature in bitsandbytes/HQQ/llama.cpp and would make the comparison about the kernels, not the quantization).
- `config.set_model("Qwen/Qwen3-1.7B")` gives a ~2.5× faster run on a smaller model.

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
| `fp8` | Qwen's FP8 checkpoint | needs compute capability ≥ 8.9; on a T4 it is dequantized to 16-bit on load |
| `torchao-int8` | optional, needs an extra install | |

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
| Drift | **mean / p99 KL divergence** vs baseline, **top-1 agreement**, perplexity | `Salesforce/wikitext` wikitext-2 test, 20 × 512 tokens. HF: our own code against saved baseline log-probs. GGUF: `llama-perplexity --kl-divergence`. Compare drift within a track only: llama-perplexity scores just the second half of each chunk, so its perplexity is lower (11.2 vs 14.4 for the same FP16 model). |
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
│   ├── run.py              # run one variant end to end; resume; record failures; CLI for the GPU-1 job
│   └── report.py           # summary.csv / .md / .png
└── results/<track>-<variant>/   metrics.json, eval.json (every reply), logs
```

**What each variant leaves in `results/<track>-<variant>/`:**

| File | Contents |
|---|---|
| `metrics.json` | **memory:** weights GB, bits/weight, peak GPU (NVML), torch peak allocated/reserved (HF) · **speed:** prefill tok/s, decode tok/s, TTFT, each with run-to-run spread and the raw runs · **workload:** GSM8K/ARC wall time, throughput, per-question latency p50/p95, llama-server TTFT p50/p95 and decode tok/s under 8 concurrent requests · **quality:** GSM8K strict/flexible, ARC, perplexity, mean/p99 KLD, top-1 agreement, tokens per answer, answers that hit the token cap · **time:** each phase (make/load, bench, tasks, KLD) and total, start/finish timestamps · **context:** GPU, driver, library versions, llama.cpp commit, run settings |
| `eval.json` | every GSM8K/ARC question: reply, correct or not, tokens, latency (+ server TTFT/decode for GGUF) |
| `run.log` | everything printed while the variant ran |
| `quant.log`, `kld.log`, `server.log` | raw llama.cpp tool output (GGUF) |
| `error.txt` + `partial.json` | only on failure: traceback, plus every metric measured before it failed |

`report.table()` merges every `metrics.json` into `results/summary.csv` / `summary.md`, adding derived columns: answers lost/gained vs baseline, and `strict_gap` (format breakage).

## 6. How to run

1. On Kaggle: **New Notebook → File → Import Notebook →** `https://github.com/jprakash-1/model-quantization-exp/blob/main/kaggle_master.ipynb`
2. Sidebar: **Accelerator = GPU T4 ×2**, **Internet = On**.
3. First time, set `N_QUESTIONS = 100` in cell 2 and run the cells interactively to catch problems early.
4. Then set it to 250 and **Save Version → Save & Run All (Commit)** to run unattended.
5. Download `results/` from the notebook output, or push it back with the last cell.
6. Re-runs resume: variants already in `results/` (committed to the repo) are skipped, and only the missing ones run. The KLD references live in `/tmp`, so in a new session the code re-runs each track's baseline first when it has anything new to measure.

What the notebook does, in order:

| Step | Cell | What happens |
|---|---|---|
| Code | 1 | clone or pull this repo, `pip install -r requirements-kaggle.txt` |
| Config | 2 | model, number of questions, both variant lists. The kernel sees only GPU 0. |
| **Data** | 3 | `data.download()` writes wikitext-2 test, GSM8K test[:N] and ARC-Challenge test[:N] to `data/` |
| Build | 5 | llama.cpp with CUDA for sm_75 → `/tmp/qlab/llama.cpp` |
| **Models** | 6 | `hf.download()`: the original, Unsloth bnb-4bit and FP8 repos → HF cache. `gg.prepare()`: F16 GGUF + imatrix + `UD-*` → `/tmp/qlab/gguf`. About 30 GB total; already-measured variants are skipped. |
| GGUF track | 8 | background job on GPU 1, makes, measures and deletes each local quant |
| HF track | 10 | in the kernel on GPU 0 |
| Report | 12–15 | wait for GPU 1, then the table, plots and lost answers |
| Cleanup | 17 | delete `/tmp/qlab` and the HF cache; `results/` stays |

Peak disk use is about 40 GB: HF cache 16 GB, GGUF downloads 12 GB, one local quant ≤ 4.3 GB, two KLD reference sets of ~3 GB each, and the llama.cpp build. Cell 6 prints `df` so you can check.

Rough time at n=250: llama.cpp build ~15 min, then both tracks in parallel. The HF track takes ~4–6 h (HQQ and bnb-int8 are the slow ones) and the GGUF track ~3 h, so **wall time is ~5–7 h**, which fits one 12 h session.

## 7. Questions the results should answer

These are in the last notebook cell too. Write the answers into `FINDINGS.md`.
1. Is 8-bit indistinguishable from the baseline? Is `bnb-int8` actually *slower* than FP16?
2. At 4-bit, which matters most: the code book, the scales, calibration, or per-layer mixing?
3. Does perplexity move under 1% while KLD and GSM8K clearly move?
4. Does GSM8K drop before ARC?
5. Do some quants break the answer *format* (strict ≪ flexible) before the math?
6. Where is the cliff for a 4B model: 3-bit or 2-bit? Do imatrix or UD move it?
7. Which formats are smaller but *not* faster (dequantization overhead)?
8. Which variants use more GPU memory than their bits suggest, and why?
9. What would you deploy on a 16 GB GPU?

## 8. Known risks

- **Library drift on Kaggle.** The notebook installs the latest `transformers` / `bitsandbytes` / `hqq`. If one breaks, that variant fails with `error.txt` and the rest continue. Pin a version in `requirements-kaggle.txt` to fix it.
- **`torchao` must match the torch version**, so it is an optional cell.
- **Batched HF generation** pads on the left. That can change a few greedy tokens compared with batch size 1. It's applied the same way to every HF variant, so comparisons within the track stay fair.
- **Parallel tracks share 4 CPUs.** GPU-bound numbers are unaffected, but HF single-stream speed can wobble a few % when `llama-quantize` runs next to it. For exact speed numbers, re-run `run.run("hf", v, force=True)` after the GGUF job ends.
- **The GGUF baseline is F16 and the HF baseline is FP16.** Each track's KLD is relative to its own baseline.
- **Kaggle's internet toggle needs a phone-verified account.**
