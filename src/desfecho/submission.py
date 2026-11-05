"""CSV id,prob,desfecho. A classe 1 sai se a probabilidade gravada for >= limiar."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from desfecho.schema import SUBMISSION_COLUMNS


def format_probability(value: float) -> str:
    number = float(np.clip(value, 0.0, 1.0))
    if not np.isfinite(number):
        raise ValueError("Probabilidade não numérica.")
    return f"{number:.6f}"


def build_submission(ids: list[str] | np.ndarray, probabilities: np.ndarray, threshold: float) -> pd.DataFrame:
    if len(ids) != len(probabilities):
        raise ValueError("IDs e probabilidades têm tamanhos diferentes.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("O limiar precisa estar entre 0 e 1.")
    probs = []
    labels = []
    for value in np.asarray(probabilities, dtype=float):
        text = format_probability(value)
        parsed = float(text)
        probs.append(parsed)
        labels.append(1 if parsed >= float(threshold) else 0)
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in ids],
            "prob": probs,
            "desfecho": labels,
        }
    )
    validate_submission(frame, threshold)
    return frame


def write_submission(frame: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")


def read_submission(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in SUBMISSION_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Colunas ausentes na submissão: {missing}")
    frame = frame[SUBMISSION_COLUMNS].copy()
    frame["id"] = frame["id"].astype(str)
    frame["prob"] = pd.to_numeric(frame["prob"], errors="coerce")
    frame["desfecho"] = pd.to_numeric(frame["desfecho"], errors="coerce")
    return frame


def validate_submission(frame: pd.DataFrame, threshold: float, *, expected_ids: list[str] | None = None) -> None:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"A submissão precisa ter as colunas {SUBMISSION_COLUMNS}, nessa ordem.")
    if frame.empty:
        raise ValueError("A submissão está vazia.")
    if frame["id"].duplicated().any():
        raise ValueError("Há IDs duplicados na submissão.")
    if frame["id"].eq("").any() or frame["id"].isna().any():
        raise ValueError("Há ID vazio na submissão.")
    if frame["prob"].isna().any() or frame["desfecho"].isna().any():
        raise ValueError("prob e desfecho não podem ter ausente.")
    if ((frame["prob"] < 0) | (frame["prob"] > 1)).any():
        raise ValueError("prob precisa estar em [0, 1].")
    labels = set(frame["desfecho"].astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError("desfecho só pode ser 0 ou 1.")
    expected = (frame["prob"].to_numpy(dtype=float) >= float(threshold)).astype(int)
    got = frame["desfecho"].to_numpy(dtype=int)
    if not np.array_equal(expected, got):
        raise ValueError("desfecho não reproduz o limiar único a partir de prob.")
    if expected_ids is not None:
        if frame["id"].tolist() != [str(i) for i in expected_ids]:
            raise ValueError("Os IDs da submissão não batem com o arquivo de entrada, na mesma ordem.")
