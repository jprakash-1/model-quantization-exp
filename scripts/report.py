"""Merge results/<variant>/* into results/summary.csv, results/summary.md and four plots."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
N_PARAMS = 2.03e9  # Qwen3-1.7B incl. embeddings (from the BF16 safetensors)
BANDWIDTH = 100e9  # Apple M3 memory bandwidth, bytes/s


def grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def read(path):
    return path.read_text() if path.exists() else ""


def peak_gb(log):
    v = grab(r"(\d+)\s+peak memory footprint", log)
    return round(v / 1e9, 2) if v else None


def speed(d):
    if (d / "bench.json").exists():  # llama-bench
        rows = json.loads(read(d / "bench.json"))
        pp = next(r["avg_ts"] for r in rows if r["n_prompt"] > 0)
        tg = next(r["avg_ts"] for r in rows if r["n_gen"] > 0)
        return pp, tg
    avg = read(d / "bench.txt").split("Averages:")[-1]  # mlx_lm.benchmark
    return grab(r"prompt_tps=([\d.]+)", avg), grab(r"generation_tps=([\d.]+)", avg)


def drift(d):
    if (d / "kld.json").exists():  # kld_mlx.py
        k = json.loads(read(d / "kld.json"))
        return k["ppl"], k["mean_kld"], k["p99_kld"], 100 * k["top1_agree"]
    t = read(d / "kld.txt")  # llama-perplexity
    ppl = grab(r"Mean PPL\(Q\)\s*:\s*([\d.]+)", t) or grab(r"Final estimate: PPL = ([\d.]+)", t)
    return ppl, grab(r"Mean\s+KLD:\s+([-\d.]+)", t), grab(r"99\.0%\s+KLD:\s+([-\d.]+)", t), grab(r"Same top p:\s+([\d.]+)", t)


def flips(ev, base):
    """Paired comparison with BF16 of the same engine: right->wrong and wrong->right on GSM8K."""
    if not base or ev is base:
        return None, None
    pairs = list(zip(base["gsm8k"], ev["gsm8k"]))
    return sum(b["flexible"] and not q["flexible"] for b, q in pairs), sum(q["flexible"] and not b["flexible"] for b, q in pairs)


def main():
    evals = {d.name: json.loads(read(d / "eval.json")) for d in RES.iterdir() if (d / "eval.json").exists()}
    rows = []
    for name, ev in sorted(evals.items()):
        d = RES / name
        engine = name.split("-")[0]
        size = int(read(d / "size_bytes") or 0)
        pp, tg = speed(d)
        ppl, kld, kld99, top1 = drift(d)
        lost, gained = flips(ev, evals.get(f"{engine}-bf16"))
        s = ev["summary"]
        rows.append({
            "variant": name, "engine": engine,
            "bpw": round(size * 8 / N_PARAMS, 2), "disk_gb": round(size / 1e9, 2),
            "peak_mem_gb": peak_gb(read(d / "server.log")),
            "pp512_tps": round(pp, 1) if pp else None, "tg128_tps": round(tg, 1) if tg else None,
            "ttft_ms": s["ttft_ms_median"],
            "ppl": round(ppl, 3) if ppl else None, "mean_kld": kld, "p99_kld": kld99,
            "top1_pct": round(top1, 2) if top1 else None,
            "gsm8k_strict": s["gsm8k_strict"], "gsm8k_flex": s["gsm8k_flexible"], "arc": s["arc"],
            "lost_vs_bf16": lost, "gained_vs_bf16": gained, "n": s["n"],
            "quant_time_s": grab(r"([\d.]+) real", read(d / "quant.log")),
            "quant_peak_gb": peak_gb(read(d / "quant.log")),
        })
    df = pd.DataFrame(rows).sort_values(["engine", "bpw"], ascending=[True, False])
    df.to_csv(RES / "summary.csv", index=False)
    (RES / "summary.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))
    plot(df)


def plot(df):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for engine, g in df.groupby("engine"):
        g = g.sort_values("bpw")
        ax[0, 0].plot(g.disk_gb, g.tg128_tps, "o-", label=engine)
        ax[0, 1].plot(g.bpw, g.mean_kld, "o-", label=engine)
        ax[1, 0].plot(g.bpw, g.gsm8k_strict, "o-", label=f"{engine} GSM8K")
        ax[1, 0].plot(g.bpw, g.arc, "s--", label=f"{engine} ARC")
        ax[1, 1].scatter(g.peak_mem_gb, g.gsm8k_flex, label=engine)
        for _, r in g.iterrows():
            ax[1, 1].annotate(r.variant.split("-", 1)[1], (r.peak_mem_gb, r.gsm8k_flex), fontsize=7)
    x = sorted(df.disk_gb.dropna())
    ax[0, 0].plot(x, [BANDWIDTH / (s * 1e9) for s in x], "k:", label="100 GB/s ÷ size")
    ax[0, 0].set(title="Decode speed vs size", xlabel="file size (GB)", ylabel="tg128 tok/s")
    ax[0, 1].set(title="Drift from BF16", xlabel="bits per weight", ylabel="mean KLD", yscale="log")
    ax[1, 0].set(title="Task accuracy vs bits", xlabel="bits per weight", ylabel="accuracy")
    ax[1, 1].set(title="Memory vs GSM8K (flexible)", xlabel="peak memory (GB)", ylabel="accuracy")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "summary.png", dpi=120)


if __name__ == "__main__":
    main()
