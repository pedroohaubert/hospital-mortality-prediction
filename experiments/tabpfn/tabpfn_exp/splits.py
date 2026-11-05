"""Índices oficiais. A fonte é desfecho.cv.official_splits."""

from __future__ import annotations

import hashlib

import numpy as np

from desfecho.config import TrainConfig
from desfecho.cv import official_splits
from tabpfn_exp.protocol import SPLIT_INDEX_SHA256


def fingerprint(y: np.ndarray, cfg: TrainConfig | None = None) -> str:
    y = np.asarray(y)
    lines = []
    for i, (_, valid_idx) in enumerate(official_splits(len(y), y, cfg)):
        body = ",".join(str(int(v)) for v in valid_idx.tolist())
        lines.append(f"{i}:{body}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def assert_official_splits(y: np.ndarray, cfg: TrainConfig | None = None) -> str:
    digest = fingerprint(y, cfg)
    if digest != SPLIT_INDEX_SHA256:
        raise RuntimeError(
            "Os folds não batem com o treino entregue. "
            f"Esperado {SPLIT_INDEX_SHA256}, obtido {digest}."
        )
    return digest
