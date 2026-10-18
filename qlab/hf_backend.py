"""Track 1: Hugging Face Transformers on GPU. Every quantization method loads through from_pretrained.

Baseline is BF16 on GPUs that support it (compute capability >= 8, e.g. A100/L4) and FP16 on a P100/T4.
"""
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qlab import config
from qlab.data import wiki_text

torch.zeros(1, device="cuda:0")  # create the CUDA context now, so it isn't counted in the first variant's memory
DTYPE = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
BASELINE = "bf16" if DTYPE == torch.bfloat16 else "fp16"


def bnb4(quant_type, double=False):
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant_type,
                              bnb_4bit_use_double_quant=double, bnb_4bit_compute_dtype=DTYPE)


def hqq(bits):
    from transformers import HqqConfig
    return HqqConfig(nbits=bits, group_size=64)


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
    "hqq4":         (None, lambda: hqq(4)),                                      # calibration-free, g64
    "hqq3":         (None, lambda: hqq(3)),
    "hqq2":         (None, lambda: hqq(2)),
    "torchao-int8": (None, torchao_int8),                                        # int8 weight-only
    "fp8":          ("{id}-FP8", None),                                          # needs compute capability >= 8.9
}


class HFModel:
    def __init__(self, variant):
        repo, make_qcfg = VARIANTS[variant]
        repo = (repo or config.MODEL_ID).format(id=config.MODEL_ID, name=config.MODEL_NAME)
        kwargs = {"quantization_config": make_qcfg()} if make_qcfg else {}

        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(config.MODEL_ID, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=DTYPE, device_map="cuda:0", **kwargs).eval()
        self.load_s = round(time.perf_counter() - t0, 1)
        self.repo = repo
        self.weights_gb = round(self.model.get_memory_footprint() / 1e9, 3)
        self.n_params = sum(p.numel() for p in self.model.parameters())  # packed for quantized models; use the baseline's

    @torch.no_grad()
    def generate(self, chats, max_tokens):
        """Greedy, batched. Returns [(text, n_new_tokens)]."""
        out = []
        for i in range(0, len(chats), config.BATCH):
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                     for c in chats[i : i + config.BATCH]]
            enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            gen = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            for row in new:
                out.append((self.tok.decode(row, skip_special_tokens=True), int((row != self.tok.pad_token_id).sum())))
            print(f"  generated {len(out)}/{len(chats)}", end="\r", flush=True)
        print()
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
            return sorted(times)[reps // 2]  # median

        prefill_s = timed(lambda: self.model(ids))
        n = config.GEN_TOKENS
        total_s = timed(lambda: self.model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False,
                                                    temperature=None, top_p=None, top_k=None,
                                                    pad_token_id=self.tok.pad_token_id))
        return {"prefill_tps": round(config.CTX / prefill_s, 1),
                "ttft_ms": round(prefill_s * 1000, 1),
                "decode_tps": round(n / (total_s - prefill_s), 1)}

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
    nll = []
    for i, chunk in enumerate(kld_chunks(m.tok)):
        lp = m.logprobs(chunk)
        torch.save(lp.half().cpu(), ref_dir() / f"{i}.pt")
        nll.append(-lp[:-1].gather(1, torch.tensor(chunk[1:], device="cuda:0")[:, None]).mean().item())
    return {"ppl": round(float(torch.tensor(nll).mean().exp()), 4), "mean_kld": 0.0, "top1_agree": 1.0}


@torch.no_grad()
def compare_to_reference(m):
    """KL(P_base || P_quant) per token, top-1 agreement, and perplexity, on the same chunks."""
    kls, agree, nll = [], [], []
    for i, chunk in enumerate(kld_chunks(m.tok)):
        ref = torch.load(ref_dir() / f"{i}.pt").to("cuda:0").float()[:-1]
        lp = m.logprobs(chunk)[:-1]
        kls.append((ref.exp() * (ref - lp)).sum(-1))
        agree.append((ref.argmax(-1) == lp.argmax(-1)).float())
        nll.append(-lp.gather(1, torch.tensor(chunk[1:], device="cuda:0")[:, None]).squeeze(1))
    kls, agree, nll = torch.cat(kls), torch.cat(agree), torch.cat(nll)
    return {"ppl": round(float(nll.mean().exp()), 4),
            "mean_kld": round(float(kls.mean()), 5),
            "p99_kld": round(float(kls.quantile(0.99)), 4),
            "top1_agree": round(float(agree.mean()), 4)}
