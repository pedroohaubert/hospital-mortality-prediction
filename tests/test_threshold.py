import numpy as np

from desfecho.threshold import select_threshold


def test_limiar_nao_zera_a_classe_negativa():
    rng = np.random.default_rng(0)
    y = np.array([1] * 180 + [0] * 120)
    scores = np.concatenate(
        [
            rng.uniform(0.2, 0.95, 180),
            rng.uniform(0.05, 0.8, 120),
        ]
    )
    choice, _ = select_threshold(y, scores, n_boot=40, seed=3, min_neg_rate=0.08, f1_margin=0.08)
    assert choice.point["taxa_prevista_0"] >= 0.05
    assert choice.point["especificidade"] > 0
    assert np.isfinite(choice.point["lr_neg"])
    assert 0 < choice.threshold < 1


def test_limiar_prefere_sensibilidade_alta_a_youden_em_separacao_clara():
    rng = np.random.default_rng(4)
    y = np.array([1] * 300 + [0] * 200)
    scores = np.concatenate([rng.normal(1.2, 0.7, 300), rng.normal(-0.4, 0.7, 200)])
    scores = 1 / (1 + np.exp(-scores))
    choice, curve = select_threshold(y, scores, n_boot=30, seed=5)
    youden = max(curve, key=lambda row: row["sensibilidade"] + row["especificidade"])
    assert choice.point["sensibilidade"] >= youden["sensibilidade"] - 1e-9
    assert choice.point["lr_neg"] <= youden["lr_neg"] + 1e-9
