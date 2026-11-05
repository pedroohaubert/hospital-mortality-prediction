"""Limiar único, escolhido só com escore out-of-fold.

A LR− melhora quando a sensibilidade sobe, mas especificidade zero é infinito
no regulamento. O piso de casos previstos como cura existe para não encostar
nisso. O piso de F1 evita um limiar que destrua o segundo desempate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from desfecho.metrics import binary_rates, stratified_indices

# Escrito antes de ver a validação. Não mexer depois de olhar o teste.
MIN_NEG_RATE = 0.08
F1_MARGIN = 0.08
FALLBACK_NEG_RATE = 0.05
FALLBACK_F1_MARGIN = 0.10


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    rule: str
    level: str
    point: dict[str, float]
    p75_lr_neg: float
    mean_lr_neg_boot: float


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    grid = np.quantile(np.asarray(scores, dtype=float), np.linspace(0.01, 0.99, 197))
    rounded = np.unique(np.round(grid, 6))
    return rounded[(rounded > 0.0) & (rounded < 1.0)]


def _curve(y: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> list[dict]:
    rows = []
    for threshold in grid:
        point = binary_rates(y, scores, float(threshold))
        point["limiar"] = float(threshold)
        rows.append(point)
    return rows


def _feasible(rows: list[dict], min_neg: float, f1_floor: float) -> list[dict]:
    kept = []
    for row in rows:
        if row["especificidade"] <= 0 or not np.isfinite(row["lr_neg"]):
            continue
        if row["taxa_prevista_0"] + 1e-12 < min_neg:
            continue
        if row["f1"] + 1e-12 < f1_floor:
            continue
        kept.append(row)
    return kept


def _tkey(value: float) -> float:
    return round(float(value), 6)


def _pick(rows: list[dict], p75: dict[float, float], boot_mean: dict[float, float]) -> dict:
    def key(row: dict) -> tuple:
        threshold = _tkey(row["limiar"])
        return (
            p75[threshold],
            boot_mean[threshold],
            -row["f1"],
            -row["especificidade"],
            -threshold,
        )

    return min(rows, key=key)


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = 400,
    seed: int = 42,
    min_neg_rate: float = MIN_NEG_RATE,
    f1_margin: float = F1_MARGIN,
) -> tuple[ThresholdChoice, list[dict]]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(scores, dtype=float)
    grid = candidate_thresholds(p)
    rows = _curve(y, p, grid)
    max_f1 = max(row["f1"] for row in rows)
    ladder = [
        (min_neg_rate, max_f1 - f1_margin, "principal"),
        (FALLBACK_NEG_RATE, max_f1 - FALLBACK_F1_MARGIN, "folga"),
        (FALLBACK_NEG_RATE, 0.0, "sem_piso_de_f1"),
    ]
    chosen_rows: list[dict] = []
    level = ""
    for neg_min, f1_floor, name in ladder:
        chosen_rows = _feasible(rows, neg_min, f1_floor)
        level = name
        if chosen_rows:
            break
    if not chosen_rows:
        finite = [row for row in rows if row["especificidade"] > 0 and np.isfinite(row["lr_neg"])]
        sens_ok = [row for row in finite if row["sensibilidade"] >= 0.95] or [
            row for row in finite if row["sensibilidade"] >= 0.90
        ]
        pool = sens_ok or finite
        if not pool:
            raise RuntimeError("Nenhum limiar com especificidade positiva.")
        best = max(pool, key=lambda row: (row["especificidade"], row["f1"], row["limiar"]))
        chosen_rows = [best]
        level = "especificidade_maxima_com_sensibilidade"

    thresholds = np.array([row["limiar"] for row in chosen_rows], dtype=float)
    boot = _bootstrap_lr(y, p, thresholds, n_boot=n_boot, seed=seed)
    p75 = {_tkey(t): float(np.quantile(boot[:, i], 0.75)) for i, t in enumerate(thresholds)}
    boot_mean = {_tkey(t): float(np.mean(boot[:, i])) for i, t in enumerate(thresholds)}
    best = _pick(chosen_rows, p75, boot_mean)
    threshold = float(f"{best['limiar']:.6f}")
    point = binary_rates(y, p, threshold)
    rule = (
        "Minimiza o p75 bootstrap da LR− entre limiares com pelo menos "
        f"{min_neg_rate:.0%} previstos como 0 e F1 a no máximo {f1_margin:.2f} "
        "do máximo F1 na mesma grade. Empate: maior F1, maior especificidade, limiar maior. "
        f"Nível aplicado: {level}."
    )
    choice = ThresholdChoice(
        threshold=threshold,
        rule=rule,
        level=level,
        point=point,
        p75_lr_neg=p75[best["limiar"]],
        mean_lr_neg_boot=boot_mean[best["limiar"]],
    )
    return choice, rows


def _bootstrap_lr(
    y: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """LR− em cada reamostra e em cada limiar. Especificidade zero vira 1e6."""
    rng = np.random.default_rng(seed)
    indices = stratified_indices(y, n_boot, rng)
    n_pos = int(np.sum(y == 1))
    n_neg = int(len(y) - n_pos)
    out = np.empty((n_boot, len(thresholds)), dtype=float)
    for i, idx in enumerate(indices):
        sample_y = y[idx]
        sample_p = scores[idx]
        pred = sample_p[:, None] >= thresholds[None, :]
        positive = sample_y[:, None] == 1
        tp = np.sum(pred & positive, axis=0)
        tn = np.sum(~pred & ~positive, axis=0)
        sens = tp / n_pos if n_pos else np.zeros(len(thresholds))
        spec = tn / n_neg if n_neg else np.zeros(len(thresholds))
        lr = np.divide(1.0 - sens, spec, out=np.full(len(thresholds), 1e6), where=spec > 0)
        out[i] = lr
    return out


def threshold_at_sensitivity(y_true: np.ndarray, scores: np.ndarray, target: float) -> dict[str, float]:
    """Maior limiar que ainda segura a sensibilidade pedida. Mais específico, sem perder o alvo."""
    grid = candidate_thresholds(scores)
    rows = _curve(np.asarray(y_true).astype(int), np.asarray(scores, dtype=float), grid)
    ok = [row for row in rows if row["sensibilidade"] + 1e-12 >= target and row["especificidade"] > 0]
    if not ok:
        best = max(rows, key=lambda row: (row["sensibilidade"], row["especificidade"]))
    else:
        best = max(ok, key=lambda row: (row["limiar"], row["especificidade"]))
    return best


def youden_and_max_f1(rows: list[dict]) -> dict[str, dict]:
    finite = [row for row in rows if row["especificidade"] > 0]
    youden = max(finite, key=lambda row: row["sensibilidade"] + row["especificidade"])
    max_f1 = max(finite, key=lambda row: (row["f1"], row["especificidade"]))
    return {"youden": youden, "max_f1": max_f1}
