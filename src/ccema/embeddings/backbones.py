"""Backbone embedding wrappers.

Defines a thin Encoder Protocol; concrete implementations:
- HuggingFaceEncoder: lazy-loads sentence-transformers for MedCPT,
  BGE-M3, all-mpnet-base-v2, MedEmbed
- OpenAIEncoder: text-embedding-3-large via the OpenAI API
- DeterministicHashEncoder: pure-Python deterministic encoder used in
  tests so no GPU/network is required
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Deterministic hash encoder for tests
# ---------------------------------------------------------------------------


@dataclass
class DeterministicHashEncoder:
    """Maps each text to a deterministic vector via SHA-256 + base remap.

    Not meaningful semantically, but identical for identical inputs and
    well-behaved for unit tests of CCE training and DRE.
    """

    name: str = "hash"
    dim: int = 128
    rng_seed: int = 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Repeat the 32-byte digest to fill `dim`, then map to [-1, 1]
            arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
            reps = -(-self.dim // arr.size)  # ceil
            arr = np.tile(arr, reps)[: self.dim]
            arr = (arr - 127.5) / 127.5
            out[i] = arr
        return out


# ---------------------------------------------------------------------------
# HuggingFace sentence-transformers encoder
# ---------------------------------------------------------------------------


@dataclass
class HuggingFaceEncoder:
    name: str
    model_id: str
    dim: int
    device: str = "cpu"
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed; pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(self.model_id, device=self.device)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        return self._model.encode(  # type: ignore[union-attr]
            list(texts), convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False
        )


# ---------------------------------------------------------------------------
# OpenAI text-embedding-3
# ---------------------------------------------------------------------------


@dataclass
class OpenAIEncoder:
    name: str = "openai-3-large"
    model_id: str = "text-embedding-3-large"
    dim: int = 3072
    api_key_env: str = "OPENAI_API_KEY"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai SDK not installed") from e
        client = OpenAI(api_key=os.environ.get(self.api_key_env))
        resp = client.embeddings.create(model=self.model_id, input=list(texts))
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory used by configs
# ---------------------------------------------------------------------------


def make_encoder(spec: str | dict) -> Encoder:
    """Build an encoder from a string id or dict spec."""
    if isinstance(spec, str):
        if spec == "hash":
            return DeterministicHashEncoder()
        if spec == "ncbi/MedCPT-Article-Encoder":
            return HuggingFaceEncoder(name="medcpt", model_id=spec, dim=768)
        if spec == "BAAI/bge-m3":
            return HuggingFaceEncoder(name="bge-m3", model_id=spec, dim=1024)
        if spec == "sentence-transformers/all-mpnet-base-v2":
            return HuggingFaceEncoder(name="mpnet", model_id=spec, dim=768)
        if spec.startswith("openai:"):
            return OpenAIEncoder(model_id=spec.removeprefix("openai:"))
        raise ValueError(f"Unknown encoder spec {spec!r}")

    return HuggingFaceEncoder(**spec)
