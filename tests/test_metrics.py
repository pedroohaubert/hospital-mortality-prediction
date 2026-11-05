import numpy as np
from sklearn.metrics import roc_auc_score

from desfecho.metrics import binary_rates, confusion, delong_auc, delong_comparison, safe_auc


def test_lr_negativa_infinita_quando_especificidade_zera():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    point = binary_rates(y, scores, threshold=0.5)
    assert point["especificidade"] == 0
    assert point["lr_neg"] == float("inf")
    assert point["sensibilidade"] == 1


def test_lr_negativa_e_f1_batem_com_a_formula():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    pred_scores = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1])
    point = binary_rates(y, pred_scores, threshold=0.5)
    counts = confusion(y, (pred_scores >= 0.5).astype(int))
    sens = counts["tp"] / (counts["tp"] + counts["fn"])
    spec = counts["tn"] / (counts["tn"] + counts["fp"])
    assert point["lr_neg"] == (1 - sens) / spec
    assert point["f1"] == (2 * counts["tp"]) / (2 * counts["tp"] + counts["fp"] + counts["fn"])


def test_delong_coincide_com_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    scores = rng.normal(size=400) + 0.7 * y
    got = delong_auc(y, scores)
    assert abs(got["auc"] - roc_auc_score(y, scores)) < 1e-9
    assert got["ic95_inf"] < got["auc"] < got["ic95_sup"]


def test_delong_pareado_ve_modelo_melhor():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 800)
    signal = rng.normal(size=800) + 1.2 * y
    weak = rng.normal(size=800) + 0.2 * y
    stat = delong_comparison(y, signal, weak)
    assert stat["delta"] > 0
    assert stat["p"] < 0.01
    assert abs(stat["auc_a"] - safe_auc(y, signal)) < 1e-9
