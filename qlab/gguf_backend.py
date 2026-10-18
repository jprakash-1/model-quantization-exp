"""Track 2: llama.cpp / GGUF on GPU. Built from source with CUDA, driven through its CLI tools.

Variant names: "F16" (baseline), "Q8_0", "Q4_K_M" (plain llama-quantize), "Q4_K_M-imat" (with Unsloth's
importance matrix), "UD-Q4_K_XL" (download Unsloth's Dynamic quant).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI

from qlab import config
from qlab.progress import Progress

BASELINE = "F16"  # T4 has no BF16, so the GGUF baseline is F16 (BF16 -> F16 is near-lossless)
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


def stream(cmd, log, bar, parse):
    """Like sh(), but reads the output as it arrives: `parse(recent_text)` returns a count that moves `bar`."""
    log.parent.mkdir(parents=True, exist_ok=True)
    out = ""
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.read1(65536), b""):
            text = chunk.decode(errors="replace")
            out += text
            f.write(text)
            n = parse(out[-4000:])
            if n is not None:
                bar.set(n)
        p.wait()
    if p.returncode:
        raise RuntimeError(f"{cmd} failed:\n{out[-3000:]}")
    bar.done()
    return out


def last_int(pattern, text):
    found = re.findall(pattern, text)
    return int(found[-1]) if found else None


def quant_pct(text):
    """llama-quantize prints `[ 123/ 398] blk.12.attn_q.weight ...` per tensor."""
    m = re.findall(r"\[\s*(\d+)/\s*(\d+)\]", text)
    return 100 * int(m[-1][0]) // int(m[-1][1]) if m else None


def ppl_chunks(text):
    """llama-perplexity: `[7]6.12,` per chunk (baseline run), or a table row `   7   6.12 ...` (KLD run)."""
    found = re.findall(r"\[(\d+)\]\d", text) + re.findall(r"(?m)^\s*(\d+)\s+\d+\.\d+", text)
    return min(max(map(int, found)), config.KLD_CHUNKS) if found else None


def cuda_driver_lib():
    """libcuda.so to link against. CMake doesn't find it on Kaggle, so we point at it: the toolkit's stub
    (same soname, libcuda.so.1, so the real driver is picked up at run time) or the driver library itself."""
    candidates = ["/usr/local/cuda/lib64/stubs/libcuda.so",
                  "/usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so",
                  "/usr/lib/x86_64-linux-gnu/libcuda.so",
                  "/usr/lib/x86_64-linux-gnu/libcuda.so.1"]
    found = [p for p in candidates if os.path.exists(p)] or sorted(glob.glob("/usr/**/libcuda.so*", recursive=True))
    if not found:
        raise RuntimeError("libcuda.so not found: is the GPU accelerator enabled?")
    return found[0]


def build():
    """~10-15 min on Kaggle's 4 CPUs. Only the four tools we use, only for this GPU's architecture (T4 = 75)."""
    if (llama_dir() / "build/bin/llama-server").exists():
        return print("llama.cpp already built")
    arch = sh("nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader").strip().replace(".", "")
    os.environ["CUDACXX"] = os.environ.get("CUDACXX") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    print(f"building llama.cpp for sm_{arch} with", sh([os.environ["CUDACXX"], "--version"]).strip().splitlines()[-1],
          "| libcuda:", cuda_driver_lib())
    config.WORK.mkdir(parents=True, exist_ok=True)
    if not (llama_dir() / ".git").exists():
        sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {llama_dir()}")
    # -D values override a stale CMake cache, and keeping build/ lets an interrupted compile resume where it stopped
    sh(f"cmake -S {llama_dir()} -B {llama_dir()}/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} "
       f"-DCUDA_cuda_driver_LIBRARY={cuda_driver_lib()} -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release")
    print("  cmake configure done; compiling (CUDA kernels are most of the time)", flush=True)
    stream(f"cmake --build {llama_dir()}/build -j {os.cpu_count()} --target llama-quantize llama-bench llama-perplexity llama-server",
           config.WORK / "llama-build.log", Progress(100, "build %", every=20), lambda t: last_int(r"\[\s*(\d+)%\]", t))
    print("  built:", sh(f"{tool('llama-server')} --version").strip().splitlines()[-1])


def gguf_dir():
    return config.WORK / "gguf"


def path(variant):
    return gguf_dir() / f"{config.MODEL_NAME}-{variant}.gguf"


def source():
    """The 16-bit GGUF every quant is made from: BF16 when Unsloth publishes it, else F16."""
    return next(p for p in (path("BF16"), path("F16")) if p.exists())


def prepare(variants=()):
    """Download the 16-bit GGUF, Unsloth's imatrix and the `UD-*` files not yet measured.
    Make the F16 baseline if the source is BF16 (needs llama.cpp built)."""
    files = list_repo_files(config.GGUF_REPO)
    src = next(f for f in (f"{config.MODEL_NAME}-BF16.gguf", f"{config.MODEL_NAME}-F16.gguf") if f in files)
    ud = [path(v).name for v in variants
          if v.startswith("UD-") and not (config.RESULTS / f"gguf-{v}" / "metrics.json").exists()]
    for f in [src, next(f for f in files if f.startswith("imatrix")), *ud]:
        print("downloading", f, flush=True)
        hf_hub_download(config.GGUF_REPO, f, local_dir=gguf_dir())
    if not path(BASELINE).exists():
        stream([tool("llama-quantize"), str(source()), str(path(BASELINE)), "F16"],
               config.WORK / "f16-convert.log", Progress(100, "BF16 -> F16 %"), quant_pct)
    print("quant source:", source().name, "| baseline:", path(BASELINE).name)


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
    cmd = [tool("llama-quantize")] + (["--imatrix", imatrix()] if imat else []) + [str(source()), str(path(variant)), qtype, "2"]  # 2 threads
    t0 = time.perf_counter()
    stream(cmd, out / "quant.log", Progress(100, f"quantize {variant} %"), quant_pct)
    return round(time.perf_counter() - t0, 1)


def bench(model):
    """llama-bench: pp512 (prefill) and tg128 (decode), all layers on GPU, 3 repetitions."""
    cmd = [tool("llama-bench"), "-m", str(model), "-ngl", "99", "-p", str(config.CTX),
           "-n", str(config.GEN_TOKENS), "-r", "3", "-o", "json"]
    rows = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)  # JSON on stdout, logs on stderr
    pp = next(r for r in rows if r["n_prompt"] > 0)
    tg = next(r for r in rows if r["n_gen"] > 0)
    return {"prefill_tps": round(pp["avg_ts"], 1), "ttft_ms": round(config.CTX / pp["avg_ts"] * 1000, 1),
            "decode_tps": round(tg["avg_ts"], 1),
            "prefill_tps_std": round(pp["stddev_ts"], 1), "decode_tps_std": round(tg["stddev_ts"], 2),
            "bench_runs": {"prefill_ns": pp.get("samples_ns"), "decode_ns": tg.get("samples_ns"),
                           "build_commit": pp.get("build_commit"), "gpu_info": pp.get("gpu_info")}}


def kld(model, variant, out):
    """llama-perplexity: the baseline saves its logits once; every quant is compared against them."""
    ref = config.WORK / f"kld-ref-{config.MODEL_NAME}.kld"
    common = [tool("llama-perplexity"), "-m", str(model), "-ngl", "99", "--kl-divergence-base", str(ref)]
    if variant == BASELINE:
        t = stream(common + ["-f", str(config.DATA / "wiki.test.raw"), "-c", str(config.CTX), "--chunks", str(config.KLD_CHUNKS)],
                   out / "kld.log", Progress(config.KLD_CHUNKS, "KLD reference chunks"), ppl_chunks)
        return {"ppl": float(re.search(r"Final estimate: PPL = ([\d.]+)", t).group(1)), "mean_kld": 0.0, "top1_agree": 1.0}
    t = stream(common + ["--kl-divergence"], out / "kld.log", Progress(config.KLD_CHUNKS, "KLD chunks"), ppl_chunks)
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
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model="default", messages=messages, temperature=0, max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        timings = (r.model_extra or {}).get("timings") or {}  # llama-server's own per-request timings
        return {"text": r.choices[0].message.content or "", "tokens": r.usage.completion_tokens,
                "latency_s": round(time.perf_counter() - t0, 3),
                "ttft_ms": timings.get("prompt_ms"), "decode_tps": timings.get("predicted_per_second")}

    def generate(self, chats, max_tokens, label="generate"):
        bar, tokens = Progress(len(chats), label), 0
        with ThreadPoolExecutor(config.PARALLEL) as pool:
            futures = [pool.submit(self._one, c, max_tokens) for c in chats]
            for f in as_completed(futures):
                tokens += f.result()["tokens"]
                bar.update(1, info=f"{tokens / (time.time() - bar.start):.0f} tok/s across {config.PARALLEL} slots")
        bar.done()
        return [f.result() for f in futures]

    def close(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()

    @staticmethod
    def buffers(log):
        """GPU memory llama.cpp reports at load: weights, KV cache, compute scratch (GB)."""
        text = log.read_text()
        mib = lambda what: sum(map(float, re.findall(rf"CUDA\d+ {what} buffer size\s*=\s*([\d.]+) MiB", text))) or None
        gb = lambda v: round(v / 1024, 3) if v else None
        return {"weights_vram_gb": gb(mib("model")), "kv_cache_gb": gb(mib("KV")), "compute_buffer_gb": gb(mib("compute"))}
