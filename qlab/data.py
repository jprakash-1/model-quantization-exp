"""Download the three open datasets into data/ as small plain files."""
import json

from datasets import load_dataset

from qlab import config


def download(n=250):
    out = config.DATA
    out.mkdir(exist_ok=True)

    # 1. wikitext-2 test: one raw text file (same as llama.cpp's wiki.test.raw)
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    (out / "wiki.test.raw").write_text("".join(wiki["text"]))

    # 2. GSM8K: question + the number after "####" in the gold answer
    gsm = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    with open(out / "gsm8k.jsonl", "w") as f:
        for row in gsm:
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"question": row["question"], "answer": answer}) + "\n")

    # 3. ARC-Challenge: question, lettered choices, gold label
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    with open(out / "arc.jsonl", "w") as f:
        for row in arc:
            choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
            f.write(json.dumps({"question": row["question"], "choices": choices, "label": row["answerKey"]}) + "\n")

    print("data ready:", sorted(p.name for p in out.iterdir()))


def load_jsonl(name, n):
    return [json.loads(line) for line in open(config.DATA / name)][:n]


def wiki_text():
    return (config.DATA / "wiki.test.raw").read_text()
