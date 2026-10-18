"""Run one variant end to end: load/build -> speed -> KLD -> tasks -> results/<track>-<variant>/.

Each finished variant writes metrics.json, so a re-run skips it (resume after a Kaggle timeout).
A failing variant writes error.txt and the loop moves on: a format that won't run on your GPU is a result too.
"""
import json
import shutil
import traceback

from qlab import config, tasks
from qlab.gpu import PeakGpuMemory


def out_dir(track, variant):
    return config.RESULTS / f"{track}-{variant}"


def save(out, metrics, evals):
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(evals, indent=1))
    (out / "metrics.json").write_text(json.dumps({**metrics, **evals["summary"]}, indent=1))
    (out / "error.txt").unlink(missing_ok=True)
    s = evals["summary"]
    print(f"  GSM8K {s['gsm8k_strict']:.3f}/{s['gsm8k_flexible']:.3f}  ARC {s['arc']:.3f}  "
          f"KLD {metrics.get('mean_kld')}  decode {metrics.get('decode_tps')} tok/s  peak {metrics.get('peak_gpu_gb')} GB")


def run_hf(variant):
    from qlab import hf_backend as hf
    m = None
    try:
        with PeakGpuMemory() as mem:  # peak = weights + batched generation, as when serving
            m = hf.HFModel(variant)
            metrics = {"track": "hf", "variant": variant, "repo": m.repo, "load_s": m.load_s,
                       "size_gb": m.weights_gb, "n_params": m.n_params, **m.bench()}
            evals = tasks.run(m.generate)
        # KLD after the sampler stops: its full-vocab log-prob buffers are not model memory
        metrics |= hf.save_reference(m) if variant == hf.BASELINE else hf.compare_to_reference(m)
    finally:
        if m:
            m.close()
    return metrics | {"peak_gpu_gb": mem.peak_gb}, evals


def run_gguf(variant, out, keep=False):
    from qlab import gguf_backend as gg
    quant_s = gg.make(variant, out)
    model = gg.path(variant)
    metrics = {"track": "gguf", "variant": variant, "quant_s": quant_s,
               "size_gb": round(model.stat().st_size / 1e9, 3),
               "n_params": gg.path(gg.BASELINE).stat().st_size // 2,  # F16 = 2 bytes per weight
               **gg.bench(model), **gg.kld(model, variant, out)}
    with PeakGpuMemory() as mem:
        server = gg.Server(model, out)
        try:
            evals = tasks.run(server.generate)
        finally:
            server.close()
    if not keep and variant not in (gg.BASELINE, "BF16"):
        model.unlink()  # rolling: free disk once measured
    return metrics | {"peak_gpu_gb": mem.peak_gb}, evals


def run(track, variant, force=False):
    out = out_dir(track, variant)
    if (out / "metrics.json").exists() and not force:
        return print(f"[skip] {track}-{variant} (done)")
    print(f"[run] {track}-{variant}")
    try:
        metrics, evals = run_hf(variant) if track == "hf" else run_gguf(variant, out)
        save(out, metrics, evals)
    except Exception:
        out.mkdir(parents=True, exist_ok=True)
        (out / "error.txt").write_text(traceback.format_exc())
        print(f"  FAILED -> {out / 'error.txt'}\n  " + traceback.format_exc().strip().splitlines()[-1])


def run_all(track, variants, force=False):
    for v in variants:
        run(track, v, force)


def cleanup():
    """Delete every big file (weights, llama.cpp build, KLD refs) and the HF cache. Results stay."""
    shutil.rmtree(config.WORK, ignore_errors=True)
    from huggingface_hub import constants
    shutil.rmtree(constants.HF_HUB_CACHE, ignore_errors=True)
    print("deleted", config.WORK, "and", constants.HF_HUB_CACHE)
