"""Score GSM8K + ARC-Challenge and time every request against a local OpenAI-compatible server.

Works with both `llama-server` and `mlx_lm.server`. Thinking mode is off for every variant.

    python scripts/eval.py --name gguf-Q4_K_M --port 8080 --limit 100
"""
import argparse
import json
import re
import statistics
import time
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
GSM_SYSTEM = "Solve the math problem step by step. End your reply with a final line of the form '#### <number>'."
ARC_SYSTEM = "Answer the multiple-choice question with only the letter of the correct choice."
NUM = r"-?\d[\d,]*\.?\d*"


def ask(client, system, user, max_tokens):
    """One greedy, streamed chat request. Returns the text and timings."""
    start = time.perf_counter()
    first = None
    text, n_tokens = "", 0
    stream = client.chat.completions.create(
        model="default_model",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    for chunk in stream:
        if chunk.usage:
            n_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            first = first or time.perf_counter()
            text += chunk.choices[0].delta.content
    end = time.perf_counter()
    first = first or end
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    decode_tps = (n_tokens - 1) / (end - first) if n_tokens > 1 and end > first else None
    return text, {"ttft_ms": (first - start) * 1000, "total_s": end - start, "tokens": n_tokens, "decode_tps": decode_tps}


def same_number(a, b):
    try:
        return abs(float(a.replace(",", "").rstrip(".")) - float(b)) < 1e-6
    except (ValueError, AttributeError):
        return False


def run_gsm8k(client, rows):
    results = []
    for i, row in enumerate(rows):
        text, t = ask(client, GSM_SYSTEM, row["question"], max_tokens=512)
        strict = re.search(rf"####\s*\$?({NUM})", text)
        numbers = re.findall(NUM, text)
        results.append({
            "i": i,
            "strict": bool(strict and same_number(strict.group(1), row["answer"])),
            "flexible": bool(numbers and same_number(numbers[-1], row["answer"])),
            "reply": text,
            **t,
        })
        print(f"gsm8k {i + 1}/{len(rows)} strict={results[-1]['strict']} {t['decode_tps'] or 0:.1f} tok/s", flush=True)
    return results


def run_arc(client, rows):
    results = []
    for i, row in enumerate(rows):
        options = "\n".join(f"{k}. {v}" for k, v in row["choices"].items())
        text, t = ask(client, ARC_SYSTEM, f"{row['question']}\n\n{options}\n\nAnswer:", max_tokens=5)
        pick = re.search(r"\b([A-E1-5])\b", text)
        results.append({"i": i, "correct": bool(pick and pick.group(1) == row["label"]), "reply": text, **t})
    return results


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--limit", type=int, default=250)
    args = p.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="local")
    load = lambda f: [json.loads(line) for line in open(ROOT / "data" / f)][: args.limit]

    ask(client, ARC_SYSTEM, "Warm-up. Reply with A.", max_tokens=2)  # first request compiles kernels; don't time it
    gsm = run_gsm8k(client, load("gsm8k.jsonl"))
    arc = run_arc(client, load("arc.jsonl"))

    summary = {
        "name": args.name,
        "n": args.limit,
        "gsm8k_strict": mean([r["strict"] for r in gsm]),
        "gsm8k_flexible": mean([r["flexible"] for r in gsm]),
        "arc": mean([r["correct"] for r in arc]),
        "ttft_ms_median": round(statistics.median(r["ttft_ms"] for r in gsm), 1),
        "decode_tps_mean": mean([r["decode_tps"] for r in gsm]),
        "gsm8k_tokens_mean": mean([r["tokens"] for r in gsm]),
    }
    out = ROOT / "results" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps({"summary": summary, "gsm8k": gsm, "arc": arc}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
