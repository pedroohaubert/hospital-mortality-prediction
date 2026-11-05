import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "tabpfn"))

from desfecho.config import TrainConfig
from desfecho.cv import official_splits
from tabpfn_exp.compare import patient_bootstrap_delta, recommend, repeat_deltas
from tabpfn_exp.device import SmokeResult, select_device
from tabpfn_exp.estimate import project_memory, project_runtime
from tabpfn_exp.loop import run_folds
from tabpfn_exp.protocol import N_FITS, SPLIT_INDEX_SHA256, TABPFN_VERSION
from tabpfn_exp.sanitize import sanitize
from tabpfn_exp.splits import assert_official_splits, fingerprint


def test_folds_oficiais_cobrem_cada_repeticao():
    y = np.array([0, 1] * 40)
    splits = official_splits(len(y), y, TrainConfig())
    assert len(splits) == N_FITS
    for repeat in range(5):
        covered = np.concatenate([valid for _, valid in splits[repeat * 5 : (repeat + 1) * 5]])
        assert sorted(covered.tolist()) == list(range(len(y)))


def test_fingerprint_do_treino_quando_o_csv_esta_local():
    path = ROOT / "data" / "final_treino_val.csv"
    if not path.is_file():
        return
    import pandas as pd

    y = pd.read_csv(path)["DESFECHO"].to_numpy()
    assert fingerprint(y) == SPLIT_INDEX_SHA256
    assert assert_official_splits(y) == SPLIT_INDEX_SHA256


def test_bootstrap_reamostra_paciente_e_nao_usa_25_folds():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    signal = rng.normal(size=200) + y
    weak = rng.normal(size=200)
    # Dois vetores, um por paciente. O terceiro eixo de fold não existe.
    delta = patient_bootstrap_delta(y, signal, weak, n_boot=80, seed=1)
    assert delta["unidade"] == "paciente"
    assert delta["n_boot"] == 80
    assert delta["delta_auc"] > 0
    repeats = np.vstack([signal + rng.normal(scale=0.05, size=200) for _ in range(5)])
    reference = np.vstack([weak for _ in range(5)])
    deltas = repeat_deltas(y, repeats, reference)
    assert len(deltas) == 5
    assert min(deltas) > 0


def test_decisao_nao_olha_fold_nem_faixa_de_empate():
    adopted = recommend(
        measured=True,
        offline_ok=True,
        bootstrap_low=0.01,
        bootstrap_high=0.03,
        delong_low=0.005,
        delong_high=0.04,
        nested_lr_candidate=0.20,
        nested_lr_reference=0.21,
        nested_f1_candidate=0.74,
        nested_f1_reference=0.74,
    )
    assert adopted["decisao"] == "adopt"
    covered = recommend(
        measured=True,
        offline_ok=True,
        bootstrap_low=-0.01,
        bootstrap_high=0.02,
        delong_low=-0.01,
        delong_high=0.02,
        nested_lr_candidate=0.1,
        nested_lr_reference=0.5,
        nested_f1_candidate=0.9,
        nested_f1_reference=0.4,
    )
    assert covered["decisao"] == "inconclusive"
    assert covered["motivo"] == "intervalo_cobre_zero"
    offline = recommend(
        measured=True,
        offline_ok=False,
        bootstrap_low=0.2,
        bootstrap_high=0.3,
        delong_low=0.2,
        delong_high=0.3,
        nested_lr_candidate=0.1,
        nested_lr_reference=0.2,
        nested_f1_candidate=0.8,
        nested_f1_reference=0.7,
    )
    assert offline["decisao"] == "reject"
    assert recommend(measured=False, offline_ok=True, bootstrap_low=1, bootstrap_high=1, delong_low=1, delong_high=1, nested_lr_candidate=0, nested_lr_reference=0, nested_f1_candidate=1, nested_f1_reference=1)["decisao"] == "inconclusive"


def test_mps_que_falha_cai_para_cpu_com_o_erro():
    def smoke(device: str) -> SmokeResult:
        if device == "mps":
            return SmokeResult(True, False, "mps", error="mps_oob")
        return SmokeResult(True, True, "cpu", seconds=0.2, peak_rss_mb=128)
    decision = select_device({"torch_installed": True, "mps_built": True, "mps_available": True}, smoke)
    assert decision.selected == "cpu"
    assert "mps" in decision.reason
    assert decision.smoke_mps.error == "mps_oob"


def test_projecao_sem_smoke_nao_inventa_tempo():
    missing = project_runtime(None)
    assert missing["status"] == "unmeasured"
    assert missing["seconds"] is None
    projected = project_runtime(0.5, smoke_rows=48, train_rows=96, n_fits=2)
    assert projected["status"] == "projected"
    assert projected["not_a_measurement"] is True
    assert projected["assumption"]
    assert project_memory(30.0)["full_fold_mb"] is None


def test_agregado_nao_leva_paciente():
    clean = sanitize({"id": ["A1"], "auc": 0.7, "probs": [0.1, 0.2], "nested": {"f1_media": 0.4}})
    assert "id" not in clean
    assert "probs" not in clean
    assert clean["auc"] == 0.7


def test_loop_retoma_fold_pronto(tmp_path: Path):
    y = np.array([0, 1] * 30)
    calls = {"n": 0}

    def predict(repeat, fold, train_idx, valid_idx):
        del repeat, fold, train_idx
        calls["n"] += 1
        return np.full(len(valid_idx), 0.3)

    cfg = TrainConfig(n_splits=2, n_repeats=1, seed=1)
    first = run_folds(y, out=tmp_path, config_hash="h", predict_fold=predict, cfg=cfg, device="cpu", cpu_override=False)
    assert calls["n"] == 2
    second = run_folds(y, out=tmp_path, config_hash="h", predict_fold=predict, cfg=cfg, device="cpu", cpu_override=False)
    assert calls["n"] == 2
    assert np.allclose(first, second)


def test_pacote_entregue_nao_depende_de_tabpfn():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "tabpfn" not in text
    manifest = json.loads((ROOT / "experiments" / "tabpfn" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "not_run"
    assert manifest["tabpfn_version"] == TABPFN_VERSION
    assert manifest["full_run_measured"] is False
    assert manifest["mps_measured"] is False
