import numpy as np

from desfecho.config import TrainConfig
from desfecho.cv import predict_proba, run_training
from desfecho.models import LogisticDesign
from desfecho.schema import AGE_COL
from desfecho.submission import build_submission, validate_submission
from tests.support import synthetic_frame


def test_mediana_da_idade_nao_ve_o_fold_de_validacao():
    frame = synthetic_frame(40, seed=1)
    train = frame.iloc[:30].copy()
    valid = frame.iloc[30:].copy()
    train[AGE_COL] = 30
    valid[AGE_COL] = 90
    design = LogisticDesign(n_knots=4).fit(train)
    assert design.medians_[AGE_COL] == 30
    transformed = design.transform(valid)
    assert np.isfinite(transformed).all()


def test_validacao_curta_e_previsao_sem_rotulo():
    frame = synthetic_frame(180, seed=8)
    cfg = TrainConfig(
        seed=8,
        n_splits=2,
        n_repeats=2,
        cat_iterations=25,
        cat_early_stopping_rounds=5,
        cat_thread_count=2,
        threshold_boot=15,
        auc_boot=20,
        spline_knots=4,
    )
    result = run_training(frame, cfg)
    oof = result["oof"]
    assert len(oof) == len(frame)
    assert oof[["p_idade", "p_logistica", "p_catboost", "p_ensemble"]].notna().all().all()
    features = frame.drop(columns=["DESFECHO"])
    probabilities = predict_proba(result["bundle"], features)
    submission = build_submission(frame["ID"].tolist(), probabilities, result["bundle"]["threshold"])
    validate_submission(submission, result["bundle"]["threshold"], expected_ids=frame["ID"].tolist())
    assert submission["desfecho"].isin([0, 1]).all()
