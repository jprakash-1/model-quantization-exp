"""Run one variant end to end and record everything about it in results/<track>-<variant>/:

    metrics.json  every number: accuracy, drift, speed, latency, memory, time per phase, environment
    eval.json     every question: reply, correct or not, tokens, latency
    run.log       everything printed while it ran
    *.log         raw tool output (GGUF: quant, kld, server)
    error.txt + partial.json   if it failed, with whatever was measured before the failure

Each finished variant writes metrics.json, so a re-run skips it (resume after a Kaggle timeout).

From the notebook: run.run_all("hf", [...]). As a background job on the second GPU:
    CUDA_VISIBLE_DEVICES=1 python -m qlab.run gguf F16 Q8_0 Q4_K_M --n 250
"""
import argparse
import contextlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback

from qlab import config, tasks
from qlab.gpu import PeakGpuMemory, physical_gpu
from qlab.progress import hms

PACKAGES = ["torch", "transformers", "accelerate", "bitsandbytes", "hqq", "torchao", "huggingface_hub", "datasets"]


def step(i, n, what):
    print(f"  {time.strftime('%H:%M:%S')}  [{i}/{n}] {what}", flush=True)


@contextlib.contextmanager
def phase(metrics, name):
    """Adds metrics['time_<name>_s'] = wall time of the block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        metrics[f"time_{name}_s"] = round(time.perf_counter() - t0, 1)


class Tee:
    """Print to the notebook/log as usual and also into the variant's run.log."""

    def __init__(self, path, stream):
        self.file, self.stream = open(path, "a"), stream

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def environment(track):
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "-i", str(physical_gpu()), "--format=csv,noheader",
                              "--query-gpu=name,driver_version,memory.total,compute_cap"],
                             capture_output=True, text=True).stdout.strip()
    env = {"gpu": gpu, "gpu_index": physical_gpu(), "python": platform.python_version()}
    for pkg in PACKAGES:
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            env[pkg] = importlib.metadata.version(pkg)
    if track == "gguf":
        from qlab import gguf_backend as gg
        env["llama.cpp"] = subprocess.run(["git", "-C", str(gg.llama_dir()), "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True).stdout.strip()
    return env


def settings():
    return {k: getattr(config, k) for k in ["MODEL_ID", "N_QUESTIONS", "KLD_CHUNKS", "CTX", "GEN_TOKENS", "BATCH", "PARALLEL"]}


def run_hf(variant, metrics):
    import torch
    from qlab import hf_backend as hf
    m = None
    try:
        torch.cuda.reset_peak_memory_stats()
        with PeakGpuMemory() as mem:  # peak = weights + speed test + batched generation, as when serving
            step(1, 4, "load (+ quantize on load)")
            with phase(metrics, "load"):
                m = hf.HFModel(variant)
            metrics |= {"repo": m.repo, "load_s": m.load_s, "size_gb": m.weights_gb, "n_params": m.n_params,
                        "dtype": str(hf.DTYPE).removeprefix("torch.")}
            step(2, 4, "speed: prefill / decode / TTFT")
            with phase(metrics, "bench"):
                metrics |= m.bench()
            print(f"    {metrics['prefill_tps']} tok/s prefill, {metrics['decode_tps']} tok/s decode, TTFT {metrics['ttft_ms']} ms")
            step(3, 4, f"tasks: GSM8K + ARC, {config.N_QUESTIONS} questions each")
            with phase(metrics, "tasks"):
                evals = tasks.run(m.generate)
        metrics |= {"peak_gpu_gb": mem.peak_gb, "gpu_used_before_gb": mem.start_gb,
                    "torch_peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
                    "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}
        # KLD after the sampler stops: its full-vocab log-prob buffers are not model memory
        step(4, 4, "drift vs baseline: KL divergence on wikitext-2")
        with phase(metrics, "kld"):
            metrics |= hf.save_reference(m) if variant == hf.BASELINE else hf.compare_to_reference(m)
    finally:
        if m:
            m.close()
    return evals


def run_gguf(variant, metrics, out, keep=False):
    from qlab import gguf_backend as gg
    step(1, 4, "make the GGUF (quantize, download, or baseline)")
    with phase(metrics, "make"):
        quant_s = gg.make(variant, out)
    model = gg.path(variant)
    metrics |= {"file": model.name, "quant_s": quant_s, "size_gb": round(model.stat().st_size / 1e9, 3),
                "n_params": gg.path(gg.BASELINE).stat().st_size // 2}  # F16 = 2 bytes per weight
    print(f"    {model.name}: {metrics['size_gb']} GB" + (f", quantized in {quant_s}s" if quant_s else ""))
    with PeakGpuMemory() as mem:  # peak = speed test + serving, same scope as the HF track
        step(2, 4, "speed: llama-bench pp512 / tg128")
        with phase(metrics, "bench"):
            metrics |= gg.bench(model)
        print(f"    {metrics['prefill_tps']} tok/s prefill, {metrics['decode_tps']} tok/s decode, TTFT {metrics['ttft_ms']} ms")
        step(3, 4, f"tasks via llama-server: GSM8K + ARC, {config.N_QUESTIONS} questions each")
        with phase(metrics, "tasks"):
            server = gg.Server(model, out)
            try:
                evals = tasks.run(server.generate)
            finally:
                server.close()
    metrics |= {"peak_gpu_gb": mem.peak_gb, "gpu_used_before_gb": mem.start_gb}
    step(4, 4, "drift vs baseline: llama-perplexity KL divergence")
    with phase(metrics, "kld"):
        metrics |= gg.kld(model, variant, out)
    if not keep and variant not in (gg.BASELINE, "BF16"):
        model.unlink()  # rolling: free disk once measured
    return evals


def run(track, variant, force=False):
    out = out_dir(track, variant)
    if (out / "metrics.json").exists() and not force:
        return print(f"[skip] {track}-{variant} (done)")
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"track": track, "variant": variant, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "settings": settings(), "env": environment(track)}
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(Tee(out / "run.log", sys.stdout)):
        try:
            evals = run_hf(variant, metrics) if track == "hf" else run_gguf(variant, metrics, out)
            metrics |= {"time_total_s": round(time.perf_counter() - t0, 1), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save(out, metrics, evals)
        except Exception:
            metrics |= {"time_total_s": round(time.perf_counter() - t0, 1), "failed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            (out / "error.txt").write_text(traceback.format_exc())
            (out / "partial.json").write_text(json.dumps(metrics, indent=1, default=str))
            print(f"  FAILED -> {out / 'error.txt'}\n  " + traceback.format_exc().strip().splitlines()[-1])


def out_dir(track, variant):
    return config.RESULTS / f"{track}-{variant}"


def save(out, metrics, evals):
    (out / "eval.json").write_text(json.dumps(evals, indent=1))
    (out / "metrics.json").write_text(json.dumps({**metrics, **evals["summary"]}, indent=1, default=str))
    (out / "error.txt").unlink(missing_ok=True)
    (out / "partial.json").unlink(missing_ok=True)
    s, m = evals["summary"], metrics
    print(f"  RESULT  GSM8K {s['gsm8k_strict']:.3f}/{s['gsm8k_flexible']:.3f}  ARC {s['arc']:.3f}  "
          f"KLD {m.get('mean_kld')}  PPL {m.get('ppl')}\n"
          f"          {m.get('size_gb')} GB weights, peak {m.get('peak_gpu_gb')} GB  |  "
          f"decode {m.get('decode_tps')} tok/s, TTFT {m.get('ttft_ms')} ms, GSM8K p50 latency {s['gsm8k_latency_p50_s']} s\n"
          f"          time: " + ", ".join(f"{k[5:-2]} {v}s" for k, v in m.items() if k.startswith("time_")))


def todo(track, variants):
    """Variants without a metrics.json yet."""
    return [v for v in variants if not (out_dir(track, v) / "metrics.json").exists()]


def reference(track):
    """The track's baseline and where it keeps the log-probs every other variant's KLD is measured against."""
    if track == "hf":
        from qlab import hf_backend as hf
        return hf.BASELINE, hf.ref_dir()
    from qlab import gguf_backend as gg
    return gg.BASELINE, gg.ref_path()


def run_all(track, variants, force=False):
    start, ran = time.time(), 0
    base, ref = reference(track)
    # the reference lives in /tmp: in a fresh session, re-run the baseline before measuring anything new against it
    redo_base = bool(todo(track, variants)) and base in variants and not ref.exists() and not force
    if redo_base:
        print(f"KLD reference missing ({ref}): re-running {base} first", flush=True)
    for i, v in enumerate(variants, 1):
        again = force or (redo_base and v == base)
        done = (out_dir(track, v) / "metrics.json").exists() and not again
        left = len(variants) - i + 1
        eta = f", ~{hms((time.time() - start) / ran * left)} left" if ran else ""
        if not done:
            print(f"\n=== {track} {i}/{len(variants)}: {v}  ({hms(time.time() - start)} into the track{eta}) ===", flush=True)
        t0 = time.time()
        run(track, v, again)
        if not done:
            ran += 1
            print(f"  {track}-{v} took {hms(time.time() - t0)}", flush=True)
    print(f"\n=== {track} track finished in {hms(time.time() - start)} ===", flush=True)


def cleanup():
    """Delete every big file (weights, llama.cpp build, KLD refs) and the HF cache. Results stay."""
    shutil.rmtree(config.WORK, ignore_errors=True)
    from huggingface_hub import constants
    shutil.rmtree(constants.HF_HUB_CACHE, ignore_errors=True)
    print("deleted", config.WORK, "and", constants.HF_HUB_CACHE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("track", choices=["hf", "gguf"])
    p.add_argument("variants", nargs="+")
    p.add_argument("--n", type=int, default=config.N_QUESTIONS)
    p.add_argument("--model", default=config.MODEL_ID)
    args = p.parse_args()
    config.N_QUESTIONS = args.n
    config.set_model(args.model)
    if args.track == "gguf":
        from qlab import gguf_backend as gg
        gg.build()
        gg.prepare(args.variants)  # no-op for files the notebook already downloaded
    run_all(args.track, args.variants)
    print("TRACK DONE", flush=True)


if __name__ == "__main__":
    main()
