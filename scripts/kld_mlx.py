"""KL divergence, top-1 agreement and perplexity of an MLX quant against the BF16 model.

Both models fit in memory at 1.7B, so we just run them side by side on the same wikitext chunks.

    python scripts/kld_mlx.py --quant models/mlx/q4 --out results/mlx-q4/kld.json
"""
import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = Path(__file__).resolve().parent.parent
CHUNKS, CTX = 20, 512


def logprobs(model, tokens):
    logits = model(mx.array(tokens)[None])[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(ROOT / "models/hf-bf16"))
    p.add_argument("--quant", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base, tokenizer = load(args.base)
    quant, _ = load(args.quant)
    ids = tokenizer.encode((ROOT / "data/wiki.test.raw").read_text())[: CHUNKS * CTX]

    kls, agree, nll_base, nll_quant = [], [], [], []
    for c in range(CHUNKS):
        chunk = ids[c * CTX : (c + 1) * CTX]
        lp_b, lp_q = logprobs(base, chunk), logprobs(quant, chunk)
        kl = (mx.exp(lp_b) * (lp_b - lp_q)).sum(axis=-1)  # KL(P_bf16 || P_quant) per position
        target = mx.array(chunk[1:])
        idx = mx.arange(CTX - 1)
        kls.append(np.array(kl[:-1]))
        agree.append(np.array(lp_b[:-1].argmax(-1) == lp_q[:-1].argmax(-1)))
        nll_base.append(-np.array(lp_b[idx, target]))
        nll_quant.append(-np.array(lp_q[idx, target]))
        print(f"chunk {c + 1}/{CHUNKS}", flush=True)

    kls = np.concatenate(kls)
    result = {
        "mean_kld": float(kls.mean()),
        "p99_kld": float(np.percentile(kls, 99)),
        "top1_agree": float(np.concatenate(agree).mean()),
        "ppl_base": math.exp(np.concatenate(nll_base).mean()),
        "ppl": math.exp(np.concatenate(nll_quant).mean()),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
