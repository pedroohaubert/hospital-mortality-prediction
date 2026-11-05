"""AUC, LR−, F1, calibração, DeLong e bootstrap.

O intervalo trata os escores como fixos e reamostra os pacientes. Não é um
intervalo do teste oculto de 1.605 linhas, e a variância da validação cruzada
não tem estimador não viesado (Bengio e Grandvalet).
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}


def binary_rates(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    """LR− = (1 − sens) / espec. Especificidade zero vira infinito, como no regulamento."""
    pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    counts = confusion(y_true, pred)
    tp, fn, fp, tn = counts["tp"], counts["fn"], counts["fp"], counts["tn"]
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    lr_neg = (1.0 - sens) / spec if spec > 0 else float("inf")
    f1_den = 2 * tp + fp + fn
    f1 = (2 * tp) / f1_den if f1_den else 0.0
    neg_rate = float(np.mean(pred == 0))
    return {
        "sensibilidade": float(sens),
        "especificidade": float(spec),
        "lr_neg": float(lr_neg),
        "f1": float(f1),
        "taxa_prevista_0": neg_rate,
        "taxa_prevista_1": float(1.0 - neg_rate),
        **{k: float(v) for k, v in counts.items()},
    }


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def fast_delong(predictions: np.ndarray, n_positive: int) -> tuple[np.ndarray, np.ndarray]:
    """Covariância de DeLong para k vetores de escore. Positivos nas primeiras colunas.

    Implementação do algoritmo rápido de Sun e Xu (2014), a partir de DeLong,
    DeLong e Clarke-Pearson (1988).
    """
    predictions = np.atleast_2d(np.asarray(predictions, dtype=np.float64))
    m = int(n_positive)
    n = predictions.shape[1] - m
    if m <= 1 or n <= 1:
        raise ValueError("DeLong precisa de pelo menos dois casos em cada classe.")
    k = predictions.shape[0]
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for row in range(k):
        tx[row] = _midrank(positive[row])
        ty[row] = _midrank(negative[row])
        tz[row] = _midrank(predictions[row])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = np.atleast_2d(sx) / m + np.atleast_2d(sy) / n
    return aucs, cov


def _ordered(y_true: np.ndarray, scores: list[np.ndarray]) -> tuple[np.ndarray, int]:
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")
    stacked = np.vstack([np.asarray(s, dtype=float)[order] for s in scores])
    return stacked, int(y.sum())


def delong_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    preds, n_pos = _ordered(y_true, [y_score])
    aucs, cov = fast_delong(preds, n_pos)
    var = float(cov.reshape(-1)[0])
    std = float(np.sqrt(max(var, 0.0)))
    auc = float(aucs[0])
    z = stats.norm.ppf(0.975)
    return {
        "auc": auc,
        "variancia": var,
        "ic95_inf": auc - z * std,
        "ic95_sup": auc + z * std,
    }


def delong_comparison(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    preds, n_pos = _ordered(y_true, [score_a, score_b])
    aucs, cov = fast_delong(preds, n_pos)
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    delta = float(aucs[0] - aucs[1])
    std = float(np.sqrt(max(var, 0.0)))
    z = delta / std if std > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    zcrit = float(stats.norm.ppf(0.975))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "delta": delta,
        "variancia_delta": var,
        "ic95_inf": delta - zcrit * std,
        "ic95_sup": delta + zcrit * std,
        "z": float(z),
        "p": p_value,
    }


def stratified_indices(y_true: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y_true).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draw_pos = rng.choice(pos, size=(n_boot, len(pos)), replace=True)
    draw_neg = rng.choice(neg, size=(n_boot, len(neg)), replace=True)
    return np.concatenate([draw_pos, draw_neg], axis=1)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    point = safe_auc(y, p)
    rng = np.random.default_rng(seed)
    indices = stratified_indices(y, n_boot, rng)
    samples = np.empty(n_boot, dtype=float)
    for i, idx in enumerate(indices):
        samples[i] = safe_auc(y[idx], p[idx])
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "auc": point,
        "ic95_inf": float(low),
        "ic95_sup": float(high),
        "n_boot": int(n_boot),
    }


def bootstrap_metric_samples(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    indices = stratified_indices(y, n_boot, rng)
    keys = ["sensibilidade", "especificidade", "lr_neg", "f1", "taxa_prevista_0"]
    store = {k: np.empty(n_boot, dtype=float) for k in keys}
    for i, idx in enumerate(indices):
        rates = binary_rates(y[idx], p[idx], threshold)
        for key in keys:
            store[key][i] = rates[key]
    return store


def calibration_summary(y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    # C grande aproxima a verossimilhança sem penalização. O slope 1 e o
    # intercepto 0 são a calibração perfeita no logit.
    fit = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    fit.fit(logit, y)
    quantiles = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    quantiles[0], quantiles[-1] = 0.0, 1.0
    # Bordas repetidas (escores muito concentrados) quebram o digitize.
    quantiles = np.unique(quantiles)
    if len(quantiles) < 3:
        bins = []
    else:
        bin_id = np.digitize(p, quantiles[1:-1], right=True)
        bins = []
        for b in range(bin_id.max() + 1):
            mask = bin_id == b
            if not np.any(mask):
                continue
            bins.append(
                {
                    "n": int(mask.sum()),
                    "prob_media": float(p[mask].mean()),
                    "obs_media": float(y[mask].mean()),
                }
            )
    ece = 0.0
    for item in bins:
        ece += (item["n"] / len(y)) * abs(item["prob_media"] - item["obs_media"])
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "brier": float(brier_score_loss(y, p)),
        "intercepto": float(fit.intercept_[0]),
        "slope": float(fit.coef_[0, 0]),
        "ece_quantis": float(ece),
        "bins": bins,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }


def mean_score(oof_repeats: np.ndarray) -> np.ndarray:
    """Média das previsões out-of-fold entre as repetições. Uma linha por paciente."""
    return np.asarray(oof_repeats, dtype=float).mean(axis=0)
