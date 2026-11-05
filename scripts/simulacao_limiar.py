"""Simulação binormal do limiar, anterior aos dados reais.

Não é resultado do modelo. Serve para ver, com AUC 0,80 e prevalência 0,35,
como a LR− e o F1 se comportam quando o limiar mira uma sensibilidade alta.
A validação de verdade está em artifacts/metrics.json.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def main() -> None:
    rng = np.random.default_rng(0)
    prevalence = 0.35
    auc = 0.80
    separation = np.sqrt(2) * norm.ppf(auc)

    def simulate(n: int) -> tuple[np.ndarray, np.ndarray]:
        labels = rng.random(n) < prevalence
        scores = rng.normal(0, 1, n) + separation * labels
        return labels, scores

    def rates(labels: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
        pred = scores >= threshold
        tp = np.sum(pred & labels)
        fn = np.sum(~pred & labels)
        fp = np.sum(pred & ~labels)
        tn = np.sum(~pred & ~labels)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        lr_neg = (1 - sens) / spec if spec > 0 else np.inf
        f1 = 2 * tp / (2 * tp + fp + fn)
        return float(sens), float(spec), float(lr_neg), float(f1)

    labels, scores = simulate(6332)
    rules: dict[str, float] = {}
    for target in (0.80, 0.90, 0.95, 0.98, 0.99, 1.0):
        if target < 1:
            rules[f"sens>={target}"] = float(np.quantile(scores[labels], 1 - target))
        else:
            rules["sens>=1.0"] = float(scores[labels].min())
    grid = np.quantile(scores, np.linspace(0.01, 0.99, 400))
    rules["Youden"] = max((sum(rates(labels, scores, t)[:2]) - 1, t) for t in grid)[1]
    rules["max F1"] = max((rates(labels, scores, t)[3], t) for t in grid)[1]

    print("regra | dev: sens spec LR- F1 | teste n=1605 (2000 sims)")
    for name, threshold in rules.items():
        sens, spec, lr_neg, f1 = rates(labels, scores, threshold)
        sample_lr = []
        sample_f1 = []
        sample_pos = []
        for _ in range(2000):
            y_test, s_test = simulate(1605)
            _, _, lr_i, f1_i = rates(y_test, s_test, threshold)
            sample_lr.append(lr_i)
            sample_f1.append(f1_i)
            sample_pos.append(float(np.mean(s_test >= threshold)))
        sample_lr = np.asarray(sample_lr)
        print(
            f"{name:10s} | {sens:.3f} {spec:.3f} {lr_neg:.3f} {f1:.3f} | "
            f"{np.median(sample_lr):.3f} [{np.quantile(sample_lr, 0.05):.3f},{np.quantile(sample_lr, 0.95):.3f}] | "
            f"F1 {np.median(sample_f1):.3f} | previstos 1 {np.median(sample_pos):.2f}"
        )
    aucs = [roc_auc_score(*simulate(1605)) for _ in range(1000)]
    print(
        "AUC teste n=1605: dp",
        round(float(np.std(aucs)), 4),
        "IC95 empirico",
        np.quantile(aucs, [0.025, 0.975]).round(3),
    )


if __name__ == "__main__":
    main()
