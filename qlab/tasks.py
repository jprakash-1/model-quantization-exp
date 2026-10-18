"""GSM8K + ARC-Challenge: prompts, scoring, and one runner that works with any backend.

A backend only has to provide `generate(list_of_chats, max_tokens, label) -> list of (text, n_tokens)`.
Decoding is greedy with thinking mode off, identical for every variant.
"""
import re
import statistics
import time

from qlab import config
from qlab.data import load_jsonl

GSM_SYSTEM = "Solve the math problem step by step. End your reply with a final line of the form '#### <number>'."
ARC_SYSTEM = "Answer the multiple-choice question with only the letter of the correct choice."
NUM = r"-?\d[\d,]*\.?\d*"


def chat(system, user):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def same_number(a, b):
    try:
        return abs(float(a.replace(",", "").rstrip(".")) - float(b)) < 1e-6
    except ValueError:
        return False


def score_gsm8k(text, gold):
    """strict = the '#### N' line is right. flexible = the last number in the reply is right."""
    strict = re.search(rf"####\s*\$?({NUM})", text)
    numbers = re.findall(NUM, text)
    return {
        "strict": bool(strict and same_number(strict.group(1), gold)),
        "flexible": bool(numbers and same_number(numbers[-1], gold)),
    }


def score_arc(text, label):
    pick = re.search(r"\b([A-E1-5])\b", text)
    return {"correct": bool(pick and pick.group(1) == label)}


def clean(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def run(generate, n=None):
    n = n or config.N_QUESTIONS
    gsm_rows, arc_rows = load_jsonl("gsm8k.jsonl", n), load_jsonl("arc.jsonl", n)

    t0 = time.perf_counter()
    gsm_out = generate([chat(GSM_SYSTEM, r["question"]) for r in gsm_rows], max_tokens=512, label="GSM8K")
    gsm_s = time.perf_counter() - t0

    arc_prompts = [
        chat(ARC_SYSTEM, r["question"] + "\n\n" + "\n".join(f"{k}. {v}" for k, v in r["choices"].items()) + "\n\nAnswer:")
        for r in arc_rows
    ]
    arc_out = generate(arc_prompts, max_tokens=5, label="ARC")

    gsm = [{"i": i, **score_gsm8k(clean(t), r["answer"]), "tokens": k, "reply": clean(t)}
           for i, (r, (t, k)) in enumerate(zip(gsm_rows, gsm_out))]
    arc = [{"i": i, **score_arc(clean(t), r["label"]), "reply": clean(t)}
           for i, (r, (t, _)) in enumerate(zip(arc_rows, arc_out))]

    mean = lambda xs: round(statistics.mean(xs), 4)
    summary = {
        "n": n,
        "gsm8k_strict": mean([g["strict"] for g in gsm]),
        "gsm8k_flexible": mean([g["flexible"] for g in gsm]),
        "arc": mean([a["correct"] for a in arc]),
        "gsm8k_tokens_mean": mean([g["tokens"] for g in gsm]),
        "gsm8k_hit_max_tokens": sum(g["tokens"] >= 512 for g in gsm),  # rambling / repetition loops
        "gsm8k_throughput_tps": round(sum(g["tokens"] for g in gsm) / gsm_s, 1),  # batched, not single-stream
    }
    return {"summary": summary, "gsm8k": gsm, "arc": arc}
