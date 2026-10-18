"""Track 1: Hugging Face Transformers on GPU. Every quantization method loads through from_pretrained.

Baseline is BF16 on GPUs that support it (compute capability >= 8, e.g. A100/L4) and FP16 on a T4 (7.5).
"""
import gc
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qlab import config
from qlab.data import wiki_text
from qlab.progress import Progress

torch.zeros(1, device="cuda:0")  # create the CUDA context now, so it isn't counted in the first variant's memory
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
BASELINE = "bf16" if DTYPE == torch.bfloat16 else "fp16"


def bnb4(quant_type, double=False):
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant_type,
                              bnb_4bit_use_double_quant=double, bnb_4bit_compute_dtype=DTYPE)


def load_hqq(repo, bits):
    """Transformers 5 no longer quantizes to HQQ on load, so call the hqq library directly:
    load 16-bit on the CPU, then swap every nn.Linear (except lm_head) for an HQQLinear on the GPU."""
    from hqq.core.quantize import BaseQuantizeConfig
    from hqq.models.hf.base import AutoHQQHFModel
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE, device_map="cpu")
    AutoHQQHFModel.quantize_model(model, quant_config=BaseQuantizeConfig(nbits=bits, group_size=64),
                                  compute_dtype=DTYPE, device="cuda:0")
    return model


def torchao_int8():
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig
    return TorchAoConfig(quant_type=Int8WeightOnlyConfig())


# name -> (repo, quantization_config factory). repo None = the original model.
VARIANTS = {
    BASELINE:       (None, None),
    "bnb-int8":     (None, lambda: BitsAndBytesConfig(load_in_8bit=True)),       # LLM.int8, outliers kept in 16-bit
    "bnb-nf4":      (None, lambda: bnb4("nf4")),                                 # QLoRA's NormalFloat4
    "bnb-nf4-dq":   (None, lambda: bnb4("nf4", double=True)),                    # + quantized scales
    "bnb-fp4":      (None, lambda: bnb4("fp4")),
    "unsloth-dyn4": ("unsloth/{name}-unsloth-bnb-4bit", None),                   # NF4 but sensitive layers left 16-bit
    "hqq4":         (None, None),                                                # calibration-free, g64 (load_hqq)
    "hqq3":         (None, None),
    "hqq2":         (None, None),
    "torchao-int8": (None, torchao_int8),                                        # int8 weight-only
    "fp8":          ("{id}-FP8", None),  # below compute capability 8.9 (T4) transformers dequantizes it to 16-bit
}
HQQ_BITS = {"hqq4": 4, "hqq3": 3, "hqq2": 2}


def repo_for(variant):
    repo = VARIANTS[variant][0] or config.MODEL_ID
    return repo.format(id=config.MODEL_ID, name=config.MODEL_NAME)


def download(variants):
    """Fetch every checkpoint up front, so `load_s` measures loading, not downloading.
    Quantize-on-load variants (bnb, hqq, torchao) all share the original repo; variants already measured are skipped."""
    from huggingface_hub import snapshot_download
    todo = [v for v in variants if not (config.RESULTS / f"hf-{v}" / "metrics.json").exists()]
    if todo and not ref_dir().exists():
        todo.append(BASELINE)  # run.run_all re-runs the baseline to rebuild the KLD reference
    repos = list(dict.fromkeys(repo_for(v) for v in todo))
    for i, repo in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] downloading {repo}", flush=True)
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


class HFModel:
    def __init__(self, variant):
        repo, make_qcfg = repo_for(variant), VARIANTS[variant][1]
        kwargs = {"quantization_config": make_qcfg()} if make_qcfg else {}

        how = (f" + quantizing on load ({type(kwargs['quantization_config']).__name__})" if kwargs
               else f" + quantizing with hqq ({HQQ_BITS[variant]}-bit)" if variant in HQQ_BITS else "")
        print(f"    loading {repo}{how}", flush=True)
        t0, before = time.perf_counter(), torch.cuda.memory_allocated()
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID, padding_side="left")
        if variant in HQQ_BITS:
            self.model = load_hqq(repo, HQQ_BITS[variant]).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE, device_map="cuda:0", **kwargs).eval()
        for mod in self.model.modules():  # pre-quantized repos bring their own compute dtype (Unsloth: bf16, emulated on a T4)
            if getattr(mod, "compute_dtype", None) is not None:
                mod.compute_dtype = DTYPE
        self.load_s = round(time.perf_counter() - t0, 1)
        self.repo = repo
        # hqq keeps scales/zeros outside the parameters, so measure what the weights take on the GPU instead
        self.weights_gb = round((torch.cuda.memory_allocated() - before if variant in HQQ_BITS
                                 else self.model.get_memory_footprint()) / 1e9, 3)
        self.n_params = sum(p.numel() for p in self.model.parameters())  # packed for quantized models; use the baseline's
        print(f"    loaded in {self.load_s}s, weights {self.weights_gb} GB", flush=True)

    @torch.no_grad()
    def generate(self, chats, max_tokens, label="generate"):
        """Greedy, batched. Returns one dict per chat: text, tokens, latency_s (= its batch's wall time)."""
        out, bar = [], Progress(len(chats), label)
        for i in range(0, len(chats), config.BATCH):
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for c in chats[i : i + config.BATCH]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            t0 = time.perf_counter()
            gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            batch_s = time.perf_counter() - t0
            new = gen[:, enc.input_ids.shape[1]:]
            for row in new:
                out.append({"text": self.tok.decode(row, skip_special_tokens=True),
                            "tokens": int((row != self.tok.pad_token_id).sum()), "latency_s": round(batch_s, 3)})
            tokens = sum(o["tokens"] for o in out)
            bar.update(len(new), info=f"{tokens / (time.time() - bar.start):.0f} tok/s at batch {config.BATCH}")
        bar.done()
        return out

    @torch.no_grad()
    def bench(self, reps=3):
        """Single stream: prefill a CTX-token prompt, then decode GEN_TOKENS tokens (EOS ignored)."""
        ids = self.tok(wiki_text()[:20000], return_tensors="pt").input_ids[:, : config.CTX].to("cuda:0")
        sync = torch.cuda.synchronize

        def timed(fn):
            fn(); sync()  # warm-up
            times = []
            for _ in range(reps):
                t0 = time.perf_counter(); fn(); sync(); times.append(time.perf_counter() - t0)
            return times

        print(f"    speed test: prefill {config.CTX} tokens, decode {config.GEN_TOKENS}, median of {reps}", flush=True)
        n = config.GEN_TOKENS
        prefill = timed(lambda: self.model(ids))
        total = timed(lambda: self.model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                                  temperature=None, top_p=None, top_k=None,
                                                  pad_token_id=self.tok.pad_token_id))
        prefill_s = statistics.median(prefill)
        decode_tps = [n / (t - prefill_s) for t in total]
        return {"prefill_tps": round(config.CTX / prefill_s, 1),
                "ttft_ms": round(prefill_s * 1000, 1),
                "decode_tps": round(statistics.median(decode_tps), 1),
                "prefill_tps_std": round(statistics.stdev(config.CTX / t for t in prefill), 1),
                "decode_tps_std": round(statistics.stdev(decode_tps), 2),
                "bench_runs": {"prefill_s": [round(t, 4) for t in prefill], "prefill_plus_decode_s": [round(t, 4) for t in total]}}

    @torch.no_grad()
    def logprobs(self, ids):
        logits = self.model(torch.tensor([ids], device="cuda:0")).logits[0].float()
        return torch.log_softmax(logits, dim=-1)

    def close(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ---- KL divergence vs the baseline ------------------------------------------------------------

def kld_chunks(tok):
    ids = tok(wiki_text(), return_tensors=None).input_ids[: config.KLD_CHUNKS * config.CTX]
    return [ids[i : i + config.CTX] for i in range(0, len(ids), config.CTX)]


def ref_dir():
    return config.WORK / f"kld-ref-{config.MODEL_NAME}"


@torch.no_grad()
def save_reference(m):
    """Baseline only: store its log-probs per chunk (fp16 on disk, ~155 MB per chunk for a 152K vocab)."""
    ref_dir().mkdir(parents=True, exist_ok=True)
    nll, bar = [], Progress(config.KLD_CHUNKS, "KLD reference chunks")
    for i, chunk in enumerate(kld_chunks(m.tok)):
        lp = m.logprobs(chunk)
        torch.save(lp.half().cpu(), ref_dir() / f"{i}.pt")
        nll.append(-lp[:-1].gather(1, torch.tensor(chunk[1:], device="cuda:0")[:, None]).mean().item())
        bar.update()
    bar.done()
    return {"ppl": round(float(torch.tensor(nll).mean().exp()), 4), "mean_kld": 0.0, "top1_agree": 1.0}


@torch.no_grad()
def compare_to_reference(m):
    """KL(P_base || P_quant) per token, top-1 agreement, and perplexity, on the same chunks."""
    kls, agree, nll, bar = [], [], [], Progress(config.KLD_CHUNKS, "KLD chunks")
    for i, chunk in enumerate(kld_chunks(m.tok)):
        ref = torch.load(ref_dir() / f"{i}.pt").to("cuda:0").float()[:-1]
        lp = m.logprobs(chunk)[:-1]
        kls.append((ref.exp() * (ref - lp)).sum(-1))
        agree.append((ref.argmax(-1) == lp.argmax(-1)).float())
        nll.append(-lp.gather(1, torch.tensor(chunk[1:], device="cuda:0")[:, None]).squeeze(1))
        bar.update(info=f"mean KLD so far {float(torch.cat(kls).mean()):.4f}")
    bar.done()
    kls, agree, nll = torch.cat(kls), torch.cat(agree), torch.cat(nll)
    return {"ppl": round(float(nll.mean().exp()), 4),
            "mean_kld": round(float(kls.mean()), 5),
            "p99_kld": round(float(kls.quantile(0.99)), 4),
            "top1_agree": round(float(agree.mean()), 4)}
