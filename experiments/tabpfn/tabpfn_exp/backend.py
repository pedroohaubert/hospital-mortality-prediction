"""Fit do TabPFN v2. Import tardio para o resto do experimento não puxar o torch."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from desfecho.data import prepare_frame
from desfecho.schema import CAT_COLS
from tabpfn_exp.protocol import CPU_SAMPLE_LIMIT_V2, MODEL_VERSION, N_ESTIMATORS


def allow_cpu_large_dataset() -> None:
    """O v2 recusa mais de 1000 linhas em CPU. O override é só esse freio."""
    os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"


def fit_predict(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    valid_x: pd.DataFrame,
    *,
    weight_path: Path,
    device: str,
    seed: int,
) -> np.ndarray:
    if device not in {"cpu", "mps"}:
        raise ValueError(f"Device não suportado neste experimento: {device}")
    if not weight_path.is_file():
        raise FileNotFoundError(f"Peso ausente: {weight_path}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    large_cpu = device == "cpu" and len(train_x) > CPU_SAMPLE_LIMIT_V2
    if large_cpu:
        allow_cpu_large_dataset()
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
    from tabpfn.settings import settings

    if large_cpu:
        settings.tabpfn.allow_cpu_large_dataset = True

    version = ModelVersion(MODEL_VERSION)
    frame_train = prepare_frame(train_x)
    frame_valid = prepare_frame(valid_x)
    cat_idx = [list(frame_train.columns).index(col) for col in CAT_COLS]
    model = TabPFNClassifier.create_default_for_version(
        version,
        model_path=str(weight_path),
        device=device,
        n_estimators=N_ESTIMATORS,
        categorical_features_indices=cat_idx,
        ignore_pretraining_limits=False,
        random_state=seed,
        n_preprocessing_jobs=1,
    )
    model.fit(frame_train, np.asarray(train_y).astype(int))
    proba = model.predict_proba(frame_valid)[:, 1]
    if not np.isfinite(proba).all():
        raise RuntimeError("O TabPFN devolveu probabilidade não finita.")
    return np.asarray(proba, dtype=float)
