"""Leitura da tabela e checagens que não usam o desfecho como feature."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from desfecho.schema import AGE_COL, BINARY_COLS, CAT_COLS, FEATURE_COLS, ID_COL, TARGET_COL

_CANONICAL = {c.lower(): c for c in [ID_COL, TARGET_COL, *FEATURE_COLS]}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aceita ID/id e DESFECHO/desfecho sem misturar o restante."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _CANONICAL:
            rename[col] = _CANONICAL[key]
    out = df.rename(columns=rename)
    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Colunas duplicadas depois da normalização: {duplicated}")
    return out


def load_table(path: Path | str, *, require_target: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = normalize_columns(frame)
    if ID_COL not in frame.columns:
        raise ValueError("A tabela precisa da coluna ID.")
    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"Colunas preditoras ausentes: {missing}")
    if require_target and TARGET_COL not in frame.columns:
        raise ValueError("A tabela de treino precisa da coluna DESFECHO.")

    frame = frame.copy()
    frame[ID_COL] = frame[ID_COL].astype(str).str.strip()
    if frame[ID_COL].eq("").any():
        raise ValueError("Há identificadores vazios.")
    if frame[ID_COL].duplicated().any():
        raise ValueError("Há identificadores duplicados.")
    if frame[ID_COL].str.contains(r"[,\n\r]", regex=True).any():
        raise ValueError("Algum ID contém vírgula ou quebra de linha e quebraria o CSV.")

    if TARGET_COL in frame.columns:
        target = pd.to_numeric(frame[TARGET_COL], errors="coerce")
        if require_target:
            if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
                raise ValueError("DESFECHO precisa ser 0 ou 1, sem ausentes, no treino.")
            frame[TARGET_COL] = target.astype(int)
        else:
            frame[TARGET_COL] = target
    return frame


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz de preditores, sem ID e sem DESFECHO.

    Categóricas viram texto estável (`26.0` e `26` colapsam para `26`).
    Numéricas ficam float, e ausente continua ausente para o modelo decidir.
    """
    frame = normalize_columns(df)
    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"Colunas preditoras ausentes: {missing}")

    out = pd.DataFrame(index=frame.index)
    for col in BINARY_COLS + [AGE_COL]:
        out[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    for col in CAT_COLS:
        out[col] = _as_category_text(frame[col])
    return out[FEATURE_COLS]


def _as_category_text(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip()
    formatted = text.copy()
    mask = numeric.notna()
    formatted.loc[mask] = numeric.loc[mask].round().astype("Int64").astype(str)
    formatted = formatted.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan, "NaN": np.nan})
    return formatted.astype("object")


def dataset_checks(df: pd.DataFrame) -> dict:
    """Descrição da base. A AUC do ID é só uma checagem de vazamento, não um modelo."""
    y = df[TARGET_COL].to_numpy()
    numeric_id = pd.to_numeric(df[ID_COL].str.replace(r"\D", "", regex=True), errors="coerce")
    if numeric_id.notna().all() and numeric_id.nunique() > 1:
        id_auc = float(roc_auc_score(y, numeric_id.to_numpy()))
    else:
        id_auc = None

    binary_offenders = {}
    for col in BINARY_COLS:
        values = pd.to_numeric(df[col], errors="coerce")
        bad = sorted(set(values.dropna().unique()) - {0.0, 1.0})
        n_missing = int(values.isna().sum())
        if bad or n_missing:
            binary_offenders[col] = {"fora_de_0_1": [float(v) for v in bad], "ausentes": n_missing}

    age = pd.to_numeric(df[AGE_COL], errors="coerce")
    return {
        "n": int(len(df)),
        "prevalencia_obito": float(np.mean(y)),
        "obitos": int(np.sum(y)),
        "curas": int(np.sum(y == 0)),
        "ausentes_preditores": int(prepare_frame(df).isna().sum().sum()),
        "id_duplicado": int(df[ID_COL].duplicated().sum()),
        "auc_id_numerico": id_auc,
        "binarias_inesperadas": binary_offenders,
        "idade_min": None if age.dropna().empty else float(age.min()),
        "idade_max": None if age.dropna().empty else float(age.max()),
        "niveis_sociodemografico4": int(df["sociodemografico4"].nunique(dropna=False)),
        "niveis_sociodemografico5": int(df["sociodemografico5"].nunique(dropna=False)),
    }
