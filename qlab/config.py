"""All knobs in one place. The notebook overrides them with `config.N_QUESTIONS = 100` etc.

Other modules read them as `config.X` at call time, so overrides take effect.
"""
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-1.7B"          # swap for "Qwen/Qwen3-4B" to repeat on a bigger model
MODEL_NAME = MODEL_ID.split("/")[-1]
GGUF_REPO = f"unsloth/{MODEL_NAME}-GGUF"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"            # small JSON results: kept in git / Kaggle output
WORK = Path("/tmp/qlab")              # big files: weights, llama.cpp build, KLD reference logits

N_QUESTIONS = 250    # per task (GSM8K, ARC-Challenge)
KLD_CHUNKS = 20      # wikitext-2 chunks for KL divergence
CTX = 512            # tokens per chunk, and the prompt length for the speed benchmark
GEN_TOKENS = 128     # tokens generated in the speed benchmark
BATCH = 16           # HF track: prompts per generate() call
PARALLEL = 8         # GGUF track: concurrent requests to llama-server


def set_model(model_id):
    """Switch model; keeps the derived repo names in sync."""
    global MODEL_ID, MODEL_NAME, GGUF_REPO
    MODEL_ID, MODEL_NAME = model_id, model_id.split("/")[-1]
    GGUF_REPO = f"unsloth/{MODEL_NAME}-GGUF"
