"""O agregado público não leva paciente, id nem probabilidade."""

from __future__ import annotations

FORBIDDEN_KEYS = {
    "id",
    "ids",
    "prob",
    "probs",
    "oof",
    "y",
    "desfecho",
    "scores",
    "predictions",
    "valid_idx",
    "train_idx",
}


def sanitize(value):
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                continue
            clean[key] = sanitize(item)
        return clean
    if isinstance(value, list):
        if len(value) > 64 and value and isinstance(value[0], (int, float)):
            return {"omitido": "vetor_longo", "n": len(value)}
        return [sanitize(item) for item in value]
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
    return value
