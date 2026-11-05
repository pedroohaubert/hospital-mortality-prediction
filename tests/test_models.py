import numpy as np

from desfecho.config import TrainConfig
from desfecho.models import LogisticDesign, make_logistic
from desfecho.schema import CAT_COLS, FEATURE_COLS, ID_COL, TARGET_COL
from tests.support import synthetic_frame


def test_id_e_desfecho_nao_entram_na_matriz():
    frame = synthetic_frame(80)
    frame["vazamento"] = frame[TARGET_COL]
    design = LogisticDesign(mode="full", n_knots=4, degree=3)
    matrix = design.fit_transform(frame)
    names = list(design.get_feature_names_out())
    assert matrix.shape[1] == len(names)
    assert ID_COL not in names
    assert TARGET_COL not in names
    assert "vazamento" not in names
    assert all(col in FEATURE_COLS or col.startswith("idade_spline_") or "[" in col for col in names)


def test_categoria_nova_cai_na_referencia_sem_quebrar():
    frame = synthetic_frame(100, seed=2)
    train = frame.iloc[:80].copy()
    valid = frame.iloc[80:].copy()
    valid[CAT_COLS[0]] = 99
    model = make_logistic(TrainConfig(seed=1), mode="full")
    model.fit(train, train["DESFECHO"])
    proba = model.predict_proba(valid)
    assert proba.shape == (len(valid), 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_logistica_de_idade_ignora_as_outras_colunas():
    frame = synthetic_frame(90, seed=3)
    model = make_logistic(TrainConfig(), mode="idade")
    model.fit(frame, frame["DESFECHO"])
    names = list(model.named_steps["design"].get_feature_names_out())
    assert names
    assert all(name.startswith("idade_spline_") for name in names)
