"""Grava e lê o pacote do modelo, as métricas e as previsões out-of-fold."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from desfecho import __version__


def save_training(directory: Path, result: dict, checks: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metrics = jsonable(result["metrics"])
    metrics["checagens"] = jsonable(checks)
    metrics["versoes"] = _versions()
    _write_json(directory / "metrics.json", metrics)
    _write_json(
        directory / "limiar.json",
        {
            "limiar": metrics["limiar"],
            "regra": metrics["limiar_regra"],
            "nivel": metrics["limiar_nivel"],
            "modelo": metrics["modelo_submissao"],
            "ponto_oof": metrics["ponto_oof"],
            "ponto_aninhado_media": metrics["ponto_aninhado_media"],
        },
    )
    _write_json(directory / "explicacao.json", jsonable(result["explanation"]))
    result["oof"].to_csv(directory / "oof.csv", index=False, float_format="%.8f")
    joblib.dump(result["bundle"], directory / "modelo.joblib", compress=3)


def load_metrics(directory: Path) -> dict:
    return json.loads((directory / "metrics.json").read_text(encoding="utf-8"))


def load_bundle(path: Path):
    return joblib.load(path)


def load_oof(directory: Path) -> pd.DataFrame:
    return pd.read_csv(directory / "oof.csv")


def load_explanation(directory: Path) -> dict:
    path = directory / "explicacao.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _versions() -> dict:
    names = ["desfecho", "numpy", "pandas", "scikit-learn", "scipy", "catboost", "matplotlib", "typer", "joblib"]
    found = {"pacote": __version__}
    for name in names:
        if name == "desfecho":
            continue
        try:
            found[name] = version(name)
        except Exception:
            found[name] = "desconhecida"
    return found
