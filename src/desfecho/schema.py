"""Nomes e papéis das colunas. O identificador não entra no modelo."""

from __future__ import annotations

ID_COL = "ID"
TARGET_COL = "DESFECHO"
AGE_COL = "sociodemografico3"
CAT_COLS = ["sociodemografico4", "sociodemografico5"]
BINARY_COLS = (
    [f"dadoclin{i}" for i in range(1, 15)]
    + ["sociodemografico1", "sociodemografico2"]
    + [f"comorbidade{i}" for i in range(1, 8)]
)
FEATURE_COLS = BINARY_COLS + [AGE_COL] + CAT_COLS

# Submissão exigida pelo regulamento, em minúsculas.
SUBMISSION_COLUMNS = ["id", "prob", "desfecho"]
