"""Verify retrieval-only loading against one previously indexed item."""
import json
import sqlite3
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_worker import Models  # noqa: E402


def cosine(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def module_gib(module):
    tensors = list(module.parameters()) + list(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors) / 2**30


database = sqlite3.connect(ROOT / ".data" / "audio-search.db")
database.row_factory = sqlite3.Row
row = database.execute("SELECT path,features_json FROM audio_items ORDER BY id LIMIT 1").fetchone()
if row is None:
    raise SystemExit("No indexed audio is available for verification")

expected = json.loads(row["features_json"])
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
models = Models()
actual = models.extract(row["path"])

for unused in ("s2mel", "semantic_codec", "campplus_model", "bigvgan", "tokenizer", "spk_matrix"):
    if hasattr(models.tts, unused):
        raise AssertionError(f"retrieval-only model unexpectedly loaded {unused}")
if hasattr(models.tts.gpt, "gpt"):
    raise AssertionError("retrieval-only model unexpectedly retained the GPT-2 generation trunk")

style_similarity = cosine(expected["style"]["embedding"], actual["style"])
emotion_similarity = cosine(expected["emotion"]["embedding"], actual["emotion"])
print(f"style cosine: {style_similarity:.8f}")
print(f"emotion cosine: {emotion_similarity:.8f}")
if min(style_similarity, emotion_similarity) < 0.999:
    raise AssertionError("retrieval-only vectors do not match the existing full-model index")
if torch.cuda.is_available():
    print(f"QwenEmotion parameters/buffers: {module_gib(models.tts.qwen_emo.model):.2f} GiB")
    print(f"W2V-BERT parameters/buffers: {module_gib(models.tts.semantic_model):.2f} GiB")
    print(f"Conditioning parameters/buffers: {module_gib(models.tts.gpt):.2f} GiB")
    print(f"Emotion prototypes: {sum(t.numel() * t.element_size() for t in models.tts.emo_matrix) / 2**30:.2f} GiB")
    print(f"CUDA allocated after extraction: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    print(f"CUDA reserved after extraction: {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
    print(f"CUDA peak allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
