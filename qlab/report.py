"""Merge results/*/metrics.json into one table, a paired flip count vs each track's baseline, and 4 plots."""
import json

import matplotlib.pyplot as plt
import pandas as pd

from qlab import config

COLUMNS = ["track", "variant", "bpw", "size_gb", "peak_gpu_gb", "prefill_tps", "ttft_ms", "decode_tps",
           "ppl", "mean_kld", "p99_kld", "top1_agree", "gsm8k_strict", "gsm8k_flexible", "arc",
           "lost_vs_base", "gained_vs_base", "gsm8k_hit_max_tokens", "gsm8k_throughput_tps", "quant_s", "load_s"]
BASELINES = {"hf": ("fp16", "bf16"), "gguf": ("f16",)}


def load():
    runs, errors = {}, {}
    for d in sorted(config.RESULTS.iterdir()):
        if (d / "metrics.json").exists():
            runs[d.name] = (json.loads((d / "metrics.json").read_text()), json.loads((d / "eval.json").read_text()))
        elif (d / "error.txt").exists():
            errors[d.name] = (d / "error.txt").read_text().strip().splitlines()[-1]
    return runs, errors


def table():
    runs, errors = load()
    base = {m["track"]: (m, ev) for m, ev in runs.values() if m["variant"] in BASELINES[m["track"]]}
    rows = []
    for m, ev in runs.values():
        b_metrics, b_eval = base.get(m["track"], (m, ev))
        pairs = list(zip(b_eval["gsm8k"], ev["gsm8k"]))  # same questions, same order
        rows.append({**m,
                     "bpw": round(m["size_gb"] * 8e9 / b_metrics["n_params"], 2),  # size / baseline params
                     "lost_vs_base": sum(b["flexible"] and not q["flexible"] for b, q in pairs),
                     "gained_vs_base": sum(q["flexible"] and not b["flexible"] for b, q in pairs)})
    df = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values(["track", "bpw"], ascending=[True, False])
    df.to_csv(config.RESULTS / "summary.csv", index=False)
    md = df.to_markdown(index=False)
    if errors:
        md += "\n\n**Failed variants**\n\n" + "\n".join(f"- `{k}`: {v}" for k, v in errors.items())
    (config.RESULTS / "summary.md").write_text(md)
    return df, errors


def plots(df):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for track, g in df.groupby("track"):
        g = g.sort_values("bpw")
        ax[0, 0].plot(g.size_gb, g.decode_tps, "o-", label=track)
        ax[0, 1].plot(g.bpw, g.mean_kld.clip(lower=1e-5), "o-", label=track)
        ax[1, 0].plot(g.bpw, g.gsm8k_strict, "o-", label=f"{track} GSM8K strict")
        ax[1, 0].plot(g.bpw, g.arc, "s--", label=f"{track} ARC")
        ax[1, 1].scatter(g.peak_gpu_gb, g.gsm8k_flexible, label=track)
        for _, r in g.iterrows():
            for a, (x, y) in ((ax[0, 1], (r.bpw, max(r.mean_kld, 1e-5))), (ax[1, 1], (r.peak_gpu_gb, r.gsm8k_flexible))):
                a.annotate(r.variant, (x, y), fontsize=7, alpha=0.8)
    ax[0, 0].set(title="Decode speed vs size (single stream)", xlabel="weights (GB)", ylabel="tok/s")
    ax[0, 1].set(title="Drift from baseline (KL divergence)", xlabel="bits per weight", ylabel="mean KLD", yscale="log")
    ax[1, 0].set(title="Which benchmark breaks first?", xlabel="bits per weight", ylabel="accuracy")
    ax[1, 1].set(title="Memory vs quality", xlabel="peak GPU memory (GB)", ylabel="GSM8K (flexible)")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS / "summary.png", dpi=120)
    return fig
