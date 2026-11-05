"""Comandos do experimento. `run` sem confirmação para no smoke."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn_exp import __version__
from tabpfn_exp.device import select_device
from tabpfn_exp.estimate import project_memory, project_runtime
from tabpfn_exp.protocol import (
    CODE_LICENSE,
    HF_FILE_URL,
    N_ESTIMATORS,
    PACKAGE_URL,
    PROTOCOL_VERSION,
    S3_FALLBACK_CHECKED,
    S3_FALLBACK_STATUS,
    SMOKE_ROWS,
    TABPFN_VERSION,
    WEIGHT_FILENAME,
    WEIGHT_LICENSE,
    WEIGHT_LICENSE_URL,
    WEIGHT_REPO,
)
from tabpfn_exp.rss import peak_rss_mb
from tabpfn_exp.sanitize import sanitize
from tabpfn_exp.weights import fetch_weight, verify_weight

HANDOFF_FILES = [
    "experiments/tabpfn/README.md",
    "experiments/tabpfn/PROTOCOLO.md",
    "experiments/tabpfn/manifest.json",
    "experiments/tabpfn/pyproject.toml",
    "experiments/tabpfn/schema/aggregate_result.schema.json",
    "docs/agents/tabpfn-experiment.md",
]


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _smoke_backend(device: str, weight: Path) -> dict:
    from desfecho.schema import BINARY_COLS, FEATURE_COLS

    from tabpfn_exp.backend import fit_predict

    rng = np.random.default_rng(0)
    n = SMOKE_ROWS
    frame = pd.DataFrame({col: rng.integers(0, 2, n) for col in BINARY_COLS})
    frame["sociodemografico3"] = rng.integers(0, 90, n)
    frame["sociodemografico4"] = rng.integers(1, 5, n)
    frame["sociodemografico5"] = rng.integers(1, 4, n)
    frame = frame[FEATURE_COLS]
    y = (frame["sociodemografico3"].to_numpy() > 45).astype(int)
    cut = n // 5
    started = time.perf_counter()
    rss_before = peak_rss_mb()
    proba = fit_predict(frame.iloc[cut:], y[cut:], frame.iloc[:cut], weight_path=weight, device=device, seed=0)
    elapsed = time.perf_counter() - started
    peak = max(peak_rss_mb(), rss_before)
    if proba.shape != (cut,) or not np.isfinite(proba).all():
        raise RuntimeError("Smoke sem probabilidade finita.")
    return {"seconds": elapsed, "peak_rss_mb": peak, "n_predicted": int(cut)}


def cmd_fetch(args: argparse.Namespace) -> int:
    result = fetch_weight(Path(args.dest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    weight = Path(args.weights)
    checked = verify_weight(weight)
    if not checked["ok"]:
        print(json.dumps(checked, ensure_ascii=False, indent=2))
        return 1
    os.environ["HF_HUB_OFFLINE"] = "1"

    def smoke(device: str):
        from tabpfn_exp.device import SmokeResult

        try:
            measured = _smoke_backend(device, weight)
        except Exception as exc:
            return SmokeResult(
                attempted=True,
                ok=False,
                device=device,
                error=f"{type(exc).__name__}: {exc}",
            )
        return SmokeResult(
            attempted=True,
            ok=True,
            device=device,
            seconds=measured["seconds"],
            peak_rss_mb=measured["peak_rss_mb"],
        )

    from tabpfn_exp.device import torch_mps_probe

    decision = select_device(torch_mps_probe(), smoke)
    runtime = project_runtime(decision.smoke_cpu.seconds if decision.selected == "cpu" else decision.smoke_mps.seconds)
    memory = project_memory(
        decision.smoke_cpu.peak_rss_mb if decision.selected == "cpu" else decision.smoke_mps.peak_rss_mb
    )
    payload = {
        "protocol": PROTOCOL_VERSION,
        "package": TABPFN_VERSION,
        "model_version": "v2",
        "status": "smoke_only",
        "measured_on_this_machine": True,
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "selected_device": decision.selected,
        "reason": decision.reason,
        "mps_built": decision.mps_built,
        "mps_available": decision.mps_available,
        "smoke_cpu_seconds": decision.smoke_cpu.seconds,
        "smoke_cpu_peak_rss_mb": decision.smoke_cpu.peak_rss_mb,
        "smoke_mps_seconds": decision.smoke_mps.seconds,
        "smoke_mps_error": decision.smoke_mps.error,
        "runtime_projection": runtime,
        "memory_projection": memory,
        "full_run": "not_started",
    }
    _dump(Path(args.out), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    path = Path(args.smoke)
    if not path.is_file():
        print(json.dumps(project_runtime(None), ensure_ascii=False, indent=2))
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    seconds = payload.get("smoke_cpu_seconds") if payload.get("selected_device") == "cpu" else payload.get("smoke_mps_seconds")
    projected = project_runtime(seconds)
    print(json.dumps({"smoke": str(path), "projection": projected, "full_run": "not_started"}, ensure_ascii=False, indent=2))
    return 0 if projected["status"] == "projected" else 1


def cmd_run(args: argparse.Namespace) -> int:
    smoke_path = Path(args.out) / "smoke.json"
    if not args.confirm_full_run:
        args.out = str(smoke_path)
        code = cmd_smoke(args)
        if code != 0:
            return code
        print(
            "Parado antes dos 25 fits. O smoke está gravado. "
            "Repita com --confirm-full-run se o tempo projetado couber na máquina."
        )
        return 0
    if not smoke_path.is_file():
        original_out = args.out
        args.out = str(smoke_path)
        code = cmd_smoke(args)
        args.out = original_out
        if code != 0:
            return code
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_only" and "selected_device" not in smoke:
        print("Smoke ausente ou inválido.")
        return 1
    return _full_run(args, smoke)


def _full_run(args: argparse.Namespace, smoke: dict) -> int:
    from desfecho.config import TrainConfig
    from desfecho.data import load_table
    from desfecho.models import fresh_catboost, make_logistic
    from tabpfn_exp.backend import fit_predict
    from tabpfn_exp.compare import (
        calibration_block,
        delong_delta,
        nested_operating_point,
        patient_bootstrap_delta,
        recommend,
        repeat_deltas,
    )
    from tabpfn_exp.loop import fold_seed, run_folds
    from desfecho.metrics import safe_auc
    from tabpfn_exp.protocol import BASELINE_ATOL, N_BOOT
    from tabpfn_exp.splits import assert_official_splits

    if not args.data:
        print("Falta --data com o CSV de treino, que não está no git.")
        return 1
    os.environ["HF_HUB_OFFLINE"] = "1"
    weight = Path(args.weights)
    checked = verify_weight(weight)
    if not checked["ok"]:
        print(json.dumps(checked, ensure_ascii=False))
        return 1
    device = smoke["selected_device"]
    if device not in {"cpu", "mps"}:
        print("Device do smoke não é cpu nem mps.")
        return 1
    frame = load_table(args.data, require_target=True)
    y = frame["DESFECHO"].to_numpy(dtype=int)
    assert_official_splits(y)
    cfg = TrainConfig()
    out = Path(args.out)
    config_hash = f"{PROTOCOL_VERSION}:{device}:{N_ESTIMATORS}:{WEIGHT_FILENAME}"

    def predict_baseline_age(repeat, fold, train_idx, valid_idx):
        del repeat, fold
        model = make_logistic(cfg, mode="idade")
        model.fit(frame.iloc[train_idx], y[train_idx])
        return model.predict_proba(frame.iloc[valid_idx])[:, 1]

    def predict_baseline_lr(repeat, fold, train_idx, valid_idx):
        del repeat, fold
        model = make_logistic(cfg, mode="full")
        model.fit(frame.iloc[train_idx], y[train_idx])
        return model.predict_proba(frame.iloc[valid_idx])[:, 1]

    def predict_baseline_cb(repeat, fold, train_idx, valid_idx):
        model = fresh_catboost(cfg, fold_seed(cfg, repeat, fold))
        model.fit(frame.iloc[train_idx], y[train_idx])
        return model.predict_proba(frame.iloc[valid_idx])[:, 1]

    def predict_tabpfn(repeat, fold, train_idx, valid_idx):
        return fit_predict(
            frame.iloc[train_idx],
            y[train_idx],
            frame.iloc[valid_idx],
            weight_path=weight,
            device=device,
            seed=fold_seed(cfg, repeat, fold),
        )

    cpu_override = device == "cpu"
    age = run_folds(y, out=out / "idade", config_hash=config_hash, predict_fold=predict_baseline_age, cfg=cfg, device="cpu", cpu_override=False)
    logistic = run_folds(y, out=out / "logistica", config_hash=config_hash, predict_fold=predict_baseline_lr, cfg=cfg, device="cpu", cpu_override=False)
    catboost = run_folds(y, out=out / "catboost", config_hash=config_hash, predict_fold=predict_baseline_cb, cfg=cfg, device="cpu", cpu_override=False)
    tabpfn = run_folds(y, out=out / "tabpfn", config_hash=config_hash, predict_fold=predict_tabpfn, cfg=cfg, device=device, cpu_override=cpu_override)
    ensemble = 0.5 * logistic + 0.5 * catboost
    three = (logistic + catboost + tabpfn) / 3.0
    baseline_path = Path(args.baseline_oof) if args.baseline_oof else None
    baseline_gap = None
    if baseline_path is not None:
        previous = pd.read_csv(baseline_path)
        if not np.array_equal(previous["id"].astype(str).to_numpy(), frame["ID"].astype(str).to_numpy()):
            raise RuntimeError("O oof.csv de referência não está na ordem dos IDs do treino.")
        baseline_gap = float(np.max(np.abs(previous["p_ensemble"].to_numpy() - ensemble.mean(axis=0))))
        if baseline_gap > BASELINE_ATOL:
            report = {
                "status": "baseline_not_reproduced",
                "max_abs_diff_ensemble": baseline_gap,
                "recommendation": {"decisao": "inconclusive", "motivo": "baseline_nao_reproduzida"},
                "full_run_completed": True,
                "patient_level_outputs": "local_only",
            }
            _dump(out / "aggregate.json", sanitize(report))
            print(json.dumps(sanitize(report), ensure_ascii=False, indent=2))
            return 2
    boot = patient_bootstrap_delta(y, tabpfn.mean(axis=0), ensemble.mean(axis=0), n_boot=N_BOOT, seed=cfg.seed)
    delong = delong_delta(y, tabpfn.mean(axis=0), ensemble.mean(axis=0))
    deltas = repeat_deltas(y, tabpfn, ensemble)
    nested_tab = nested_operating_point(
        y, tabpfn, n_boot=max(80, cfg.threshold_boot // 2), seed=cfg.seed, min_neg_rate=cfg.min_neg_rate, f1_margin=cfg.f1_margin
    )
    nested_ref = nested_operating_point(
        y, ensemble, n_boot=max(80, cfg.threshold_boot // 2), seed=cfg.seed, min_neg_rate=cfg.min_neg_rate, f1_margin=cfg.f1_margin
    )
    decision = recommend(
        measured=True,
        offline_ok=True,
        bootstrap_low=float(boot["ic95_inf"]),
        bootstrap_high=float(boot["ic95_sup"]),
        delong_low=float(delong["ic95_inf"]),
        delong_high=float(delong["ic95_sup"]),
        nested_lr_candidate=nested_tab["lr_neg_media"],
        nested_lr_reference=nested_ref["lr_neg_media"],
        nested_f1_candidate=nested_tab["f1_media"],
        nested_f1_reference=nested_ref["f1_media"],
    )
    report = {
        "status": "complete",
        "protocol": PROTOCOL_VERSION,
        "device": device,
        "device_reason": smoke.get("reason"),
        "hardware": smoke.get("hardware"),
        "baseline_max_abs_diff": baseline_gap,
        "primary": {
            "bootstrap": boot,
            "delong": delong,
            "repeat_deltas": deltas,
            "repeat_delta_min": float(np.min(deltas)),
            "repeat_delta_max": float(np.max(deltas)),
            "repeat_delta_mean": float(np.mean(deltas)),
            "repeat_note": "Cinco diferenças dependentes. Não é intervalo de confiança.",
        },
        "nested_tabpfn": {k: v for k, v in nested_tab.items() if k != "por_repeticao"},
        "nested_ensemble": {k: v for k, v in nested_ref.items() if k != "por_repeticao"},
        "calibration_tabpfn": calibration_block(y, tabpfn.mean(axis=0)),
        "calibration_ensemble": calibration_block(y, ensemble.mean(axis=0)),
        "secondary_three_mean_auc_delta": float(
            safe_auc(y, three.mean(axis=0)) - safe_auc(y, ensemble.mean(axis=0))
        ),
        "secondary_note": "A média dos três não entra na decisão.",
        "recommendation": decision,
        "patient_level_outputs": "local_only",
    }
    # O exploratório fica no disco local, fora do agregado sanitizado que a decisão usa.
    local = dict(report)
    local["exploratory_threshold_tabpfn"] = nested_tab["exploratorio"]
    local["exploratory_threshold_ensemble"] = nested_ref["exploratorio"]
    _dump(out / "aggregate_local.json", sanitize(local))
    public = dict(report)
    public["nested_tabpfn"].pop("exploratorio", None)
    public["nested_ensemble"].pop("exploratorio", None)
    _dump(out / "aggregate.json", sanitize(public))
    print(json.dumps(sanitize(public), ensure_ascii=False, indent=2))
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in HANDOFF_FILES:
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(rel)
            archive.write(path, arcname=rel)
        base = root / "experiments" / "tabpfn" / "tabpfn_exp"
        for path in sorted(base.rglob("*.py")):
            archive.write(path, arcname=str(path.relative_to(root)))
        archive.writestr(
            "LEIA.txt",
            "Pacote sem CSV, sem peso e sem previsão por paciente.\n"
            f"Código {CODE_LICENSE}. Pesos {WEIGHT_LICENSE} em {WEIGHT_LICENSE_URL}.\n"
            f"Baixar só com tabpfn-exp fetch. URL {HF_FILE_URL}\n",
        )
    print(dest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabpfn-exp", description=f"Experimento TabPFN v2 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--dest", required=True)
    fetch.set_defaults(func=cmd_fetch)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--weights", required=True)
    smoke.add_argument("--out", required=True)
    smoke.set_defaults(func=cmd_smoke)
    estimate = sub.add_parser("estimate")
    estimate.add_argument("--smoke", required=True)
    estimate.set_defaults(func=cmd_estimate)
    run = sub.add_parser("run")
    run.add_argument("--data", default=None)
    run.add_argument("--weights", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--baseline-oof", default=None)
    run.add_argument("--confirm-full-run", action="store_true")
    run.set_defaults(func=cmd_run)
    handoff = sub.add_parser("handoff")
    handoff.add_argument("--root", default=".")
    handoff.add_argument("--out", required=True)
    handoff.set_defaults(func=cmd_handoff)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.func(args))


# Referências usadas no manifesto impresso por quem lê o módulo.
_MANIFEST_HINT = {
    "package_url": PACKAGE_URL,
    "weight_repo": WEIGHT_REPO,
    "s3_fallback": S3_FALLBACK_CHECKED,
    "s3_status": S3_FALLBACK_STATUS,
    "weight_license": WEIGHT_LICENSE,
    "code_license": CODE_LICENSE,
}


if __name__ == "__main__":
    main()
