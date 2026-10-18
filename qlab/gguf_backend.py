"""Track 2: llama.cpp / GGUF on GPU. Built from source with CUDA, driven through its CLI tools.

Variant names: "f16" (baseline), "Q8_0", "Q4_K_M" (plain llama-quantize), "Q4_K_M-imat" (with Unsloth's
importance matrix), "UD-Q4_K_XL" (download Unsloth's Dynamic quant).
"""
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI

from qlab import config

BASELINE = "f16"  # a T4 has no fast BF16 path, so the GGUF baseline is F16 (BF16 -> F16 is near-lossless)
PORT = 8080


def llama_dir():
    return config.WORK / "llama.cpp"


def tool(name):
    return str(llama_dir() / "build" / "bin" / name)


def sh(cmd, log=None):
    """Run a command; keep its full output in `log` (or print it on failure)."""
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"{cmd} failed:\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout + r.stderr


def build():
    """~10-15 min on Kaggle's 4 CPUs. Only the four tools we use, only for this GPU's architecture."""
    if (llama_dir() / "build/bin/llama-server").exists():
        return print("llama.cpp already built")
    major, minor = torch.cuda.get_device_capability(0)
    config.WORK.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {llama_dir()}")
    sh(f"cmake -S {llama_dir()} -B {llama_dir()}/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={major}{minor} "
       f"-DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release")
    sh(f"cmake --build {llama_dir()}/build -j {os.cpu_count()} --target llama-quantize llama-bench llama-perplexity llama-server")
    print(sh(f"{tool('llama-server')} --version").strip().splitlines()[-1])


def gguf_dir():
    return config.WORK / "gguf"


def path(variant):
    return gguf_dir() / f"{config.MODEL_NAME}-{variant}.gguf"


def prepare():
    """Download the BF16 GGUF (quantization source) + Unsloth's imatrix, and make the F16 baseline."""
    files = list_repo_files(config.GGUF_REPO)
    for f in [f"{config.MODEL_NAME}-BF16.gguf", next(f for f in files if f.startswith("imatrix"))]:
        hf_hub_download(config.GGUF_REPO, f, local_dir=gguf_dir())
    if not path("f16").exists():
        sh([tool("llama-quantize"), str(path("BF16")), str(path("f16")), "F16"])


def imatrix():
    return str(next(gguf_dir().glob("imatrix*")))


def make(variant, out):
    """Produce the variant's GGUF file. Returns quantization wall time (s), or None if downloaded."""
    if variant == BASELINE or path(variant).exists():
        return None
    if variant.startswith("UD-"):
        hf_hub_download(config.GGUF_REPO, path(variant).name, local_dir=gguf_dir())
        return None
    qtype, imat = variant.removesuffix("-imat"), variant.endswith("-imat")
    cmd = [tool("llama-quantize")] + (["--imatrix", imatrix()] if imat else []) + [str(path("BF16")), str(path(variant)), qtype]
    t0 = time.perf_counter()
    sh(cmd, log=out / "quant.log")
    return round(time.perf_counter() - t0, 1)


def bench(model):
    """llama-bench: pp512 (prefill) and tg128 (decode), all layers on GPU, 3 repetitions."""
    cmd = [tool("llama-bench"), "-m", str(model), "-ngl", "99", "-p", str(config.CTX),
           "-n", str(config.GEN_TOKENS), "-r", "3", "-o", "json"]
    rows = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)  # JSON on stdout, logs on stderr
    pp =next(r["avg_ts"] for r in rows if r["n_prompt"] > 0)
    tg = next(r["avg_ts"] for r in rows if r["n_gen"] > 0)
    return {"prefill_tps": round(pp, 1), "ttft_ms": round(config.CTX / pp * 1000, 1), "decode_tps": round(tg, 1)}


def kld(model, variant, out):
    """llama-perplexity: the baseline saves its logits once; every quant is compared against them."""
    ref = config.WORK / f"kld-ref-{config.MODEL_NAME}.kld"
    common = [tool("llama-perplexity"), "-m", str(model), "-ngl", "99", "--kl-divergence-base", str(ref)]
    if variant == BASELINE:
        t = sh(common + ["-f", str(config.DATA / "wiki.test.raw"), "-c", str(config.CTX),
                         "--chunks", str(config.KLD_CHUNKS)], log=out / "kld.log")
        return {"ppl": float(re.search(r"Final estimate: PPL = ([\d.]+)", t).group(1)), "mean_kld": 0.0, "top1_agree": 1.0}
    t = sh(common + ["--kl-divergence"], log=out / "kld.log")
    grab = lambda p: float(re.search(p, t).group(1))
    return {"ppl": grab(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"),
            "mean_kld": grab(r"Mean\s+KLD:\s+([-\d.]+)"),
            "p99_kld": grab(r"99\.0%\s+KLD:\s+([-\d.]+)"),
            "top1_agree": round(grab(r"Same top p:\s+([\d.]+)") / 100, 4)}


class Server:
    """llama-server with PARALLEL slots; generate() fans requests out over them."""

    def __init__(self, model, out):
        self.log = open(out / "server.log", "w")
        self.proc = subprocess.Popen(
            [tool("llama-server"), "-m", str(model), "-ngl", "99", "--port", str(PORT), "--jinja",
             "-np", str(config.PARALLEL), "-c", str(1024 * config.PARALLEL)],  # 1024 tokens per slot: prompt + 512 out
            stdout=self.log, stderr=subprocess.STDOUT)
        self.client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="local", timeout=600)
        for _ in range(300):
            try:
                self.client.models.list()
                return
            except Exception:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited, see {out / 'server.log'}")
                time.sleep(1)
        raise TimeoutError("llama-server did not start")

    def _one(self, messages, max_tokens):
        r = self.client.chat.completions.create(
            model="default", messages=messages, temperature=0, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or "", r.usage.completion_tokens

    def generate(self, chats, max_tokens):
        with ThreadPoolExecutor(config.PARALLEL) as pool:
            return list(pool.map(lambda c: self._one(c, max_tokens), chats))

    def close(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()
