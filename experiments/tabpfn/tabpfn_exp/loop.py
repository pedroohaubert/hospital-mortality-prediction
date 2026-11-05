"""Folds retomáveis. O log não guarda id, feature nem credencial."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from desfecho.config import TrainConfig
from desfecho.cv import official_splits


def fold_seed(cfg: TrainConfig, repeat: int, fold: int) -> int:
    return cfg.seed + 10007 * repeat + 17 * fold


def _log_path(out: Path, repeat: int, fold: int) -> Path:
    return out / "folds" / f"r{repeat}_f{fold}.json"


def _done(path: Path, config_hash: str) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "done" and payload.get("config_hash") == config_hash


def run_folds(
    y: np.ndarray,
    *,
    out: Path,
    config_hash: str,
    predict_fold: Callable[[int, int, np.ndarray, np.ndarray], np.ndarray],
    cfg: TrainConfig | None = None,
    device: str,
    cpu_override: bool,
) -> np.ndarray:
    """Preenche um array (repetições, pacientes). Retoma o que já está `done`."""
    cfg = cfg or TrainConfig()
    y = np.asarray(y).astype(int)
    splits = official_splits(len(y), y, cfg)
    scores = np.full((cfg.n_repeats, len(y)), np.nan)
    cache = out / "oof_repeats.npy"
    if cache.is_file():
        previous = np.load(cache)
        if previous.shape == scores.shape:
            scores = previous
    (out / "folds").mkdir(parents=True, exist_ok=True)
    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        repeat, fold = divmod(fold_id, cfg.n_splits)
        path = _log_path(out, repeat, fold)
        if _done(path, config_hash) and np.isfinite(scores[repeat, valid_idx]).all():
            continue
        try:
            predicted = np.asarray(predict_fold(repeat, fold, train_idx, valid_idx), dtype=float)
            if predicted.shape != (len(valid_idx),) or not np.isfinite(predicted).all():
                raise RuntimeError("Previsão do fold incompleta.")
            scores[repeat, valid_idx] = predicted
            record = {
                "repeat": repeat,
                "fold": fold,
                "status": "done",
                "config_hash": config_hash,
                "device": device,
                "cpu_large_dataset_override": cpu_override,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
            }
        except Exception as exc:
            record = {
                "repeat": repeat,
                "fold": fold,
                "status": "failed",
                "config_hash": config_hash,
                "device": device,
                "cpu_large_dataset_override": cpu_override,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "error": f"{type(exc).__name__}: {exc}",
            }
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            np.save(cache, scores)
            raise
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        np.save(cache, scores)
    if not np.isfinite(scores).all():
        raise RuntimeError("Ficou fold sem previsão.")
    return scores
