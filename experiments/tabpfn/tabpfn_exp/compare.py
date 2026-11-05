"""Contraste pareado no paciente. Fold não é unidade amostral."""

from __future__ import annotations

import numpy as np

from desfecho.metrics import binary_rates, delong_comparison, safe_auc, stratified_indices
from desfecho.threshold import select_threshold
from tabpfn_exp.protocol import (
    F1_WORSE_TOLERANCE,
    LR_NEG_WORSE_TOLERANCE,
    N_BOOT,
    PRIMARY_CANDIDATE,
    REFERENCE,
    SEED,
)


def repeat_deltas(y: np.ndarray, candidate: np.ndarray, reference: np.ndarray) -> list[float]:
    """Uma diferença de AUC por repetição. As cinco não são uma amostra iid."""
    y = np.asarray(y).astype(int)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("Os escores por repetição precisam ter o mesmo formato (R, N).")
    deltas = []
    for row_c, row_r in zip(candidate, reference, strict=True):
        deltas.append(safe_auc(y, row_c) - safe_auc(y, row_r))
    return deltas


def patient_bootstrap_delta(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict[str, float | str | int]:
    """IC da diferença de AUC reamostrando pacientes, com os escores fixos.

    Não reamostra folds. Não inclui a variância de reajustar o modelo.
    """
    y = np.asarray(y).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)
    point = safe_auc(y, score_a) - safe_auc(y, score_b)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i, idx in enumerate(stratified_indices(y, n_boot, rng)):
        draws[i] = safe_auc(y[idx], score_a[idx]) - safe_auc(y[idx], score_b[idx])
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_auc": float(point),
        "ic95_inf": float(low),
        "ic95_sup": float(high),
        "n_boot": int(n_boot),
        "unidade": "paciente",
        "tipo": "bootstrap_condicional_nos_escores",
    }


def delong_delta(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float | str]:
    stat = delong_comparison(y, score_a, score_b)
    return {
        "delta_auc": float(stat["delta"]),
        "ic95_inf": float(stat["ic95_inf"]),
        "ic95_sup": float(stat["ic95_sup"]),
        "tipo": "delong_escores_fixos",
    }


def _side(low: float, high: float) -> str:
    if low > 0:
        return "acima"
    if high < 0:
        return "abaixo"
    return "cobre"


def recommend(
    *,
    measured: bool,
    offline_ok: bool,
    bootstrap_low: float | None,
    bootstrap_high: float | None,
    delong_low: float | None,
    delong_high: float | None,
    nested_lr_candidate: float | None,
    nested_lr_reference: float | None,
    nested_f1_candidate: float | None,
    nested_f1_reference: float | None,
) -> dict[str, str]:
    """Adota, rejeita ou fica inconclusivo. Não olha fold nem faixa de 0,010."""
    if not measured:
        return {"decisao": "inconclusive", "motivo": "nao_medido"}
    if not offline_ok:
        return {"decisao": "reject", "motivo": "offline_ou_checksum"}
    if None in (
        bootstrap_low,
        bootstrap_high,
        delong_low,
        delong_high,
        nested_lr_candidate,
        nested_lr_reference,
        nested_f1_candidate,
        nested_f1_reference,
    ):
        return {"decisao": "inconclusive", "motivo": "faltam_intervalos_ou_ponto_aninhado"}
    boot = _side(float(bootstrap_low), float(bootstrap_high))
    delong = _side(float(delong_low), float(delong_high))
    if boot != delong:
        return {"decisao": "inconclusive", "motivo": "bootstrap_e_delong_discordam"}
    if boot == "cobre":
        return {"decisao": "inconclusive", "motivo": "intervalo_cobre_zero"}
    if boot == "abaixo":
        return {"decisao": "reject", "motivo": "intervalo_de_auc_abaixo_de_zero"}
    if float(nested_lr_candidate) > float(nested_lr_reference) + LR_NEG_WORSE_TOLERANCE:
        return {"decisao": "inconclusive", "motivo": "lr_neg_pior_que_a_tolerancia"}
    if float(nested_f1_candidate) < float(nested_f1_reference) - F1_WORSE_TOLERANCE:
        return {"decisao": "inconclusive", "motivo": "f1_pior_que_a_tolerancia"}
    return {
        "decisao": "adopt",
        "motivo": "intervalo_acima_de_zero_e_ponto_dentro_da_tolerancia",
        "contraste": f"{PRIMARY_CANDIDATE}_menos_{REFERENCE}",
    }


def nested_operating_point(
    y: np.ndarray,
    repeats: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    min_neg_rate: float,
    f1_margin: float,
) -> dict:
    """Limiar escolhido nas outras repetições e medido na que ficou de fora.

    O corte na média de todas as repetições fica em `exploratorio` e não entra
    na decisão.
    """
    y = np.asarray(y).astype(int)
    repeats = np.asarray(repeats, dtype=float)
    held_rows = []
    for held in range(repeats.shape[0]):
        selector = np.mean(np.delete(repeats, held, axis=0), axis=0)
        evaluation = repeats[held]
        choice, _ = select_threshold(
            y,
            selector,
            n_boot=n_boot,
            seed=seed + held + 1,
            min_neg_rate=min_neg_rate,
            f1_margin=f1_margin,
        )
        point = binary_rates(y, evaluation, choice.threshold)
        point["repeticao"] = held
        point["limiar"] = choice.threshold
        held_rows.append(point)
    mean_score = repeats.mean(axis=0)
    exploratory_choice, _ = select_threshold(
        y,
        mean_score,
        n_boot=n_boot,
        seed=seed,
        min_neg_rate=min_neg_rate,
        f1_margin=f1_margin,
    )
    exploratory = binary_rates(y, mean_score, exploratory_choice.threshold)
    exploratory["limiar"] = exploratory_choice.threshold
    exploratory["papel"] = "exploratorio_mesmo_oof"
    return {
        "papel": "confirmatorio",
        "por_repeticao": held_rows,
        "lr_neg_media": float(np.mean([row["lr_neg"] for row in held_rows])),
        "f1_media": float(np.mean([row["f1"] for row in held_rows])),
        "sensibilidade_media": float(np.mean([row["sensibilidade"] for row in held_rows])),
        "especificidade_media": float(np.mean([row["especificidade"] for row in held_rows])),
        "exploratorio": exploratory,
    }


def calibration_block(y: np.ndarray, score: np.ndarray) -> dict:
    from desfecho.metrics import calibration_summary

    summary = calibration_summary(y, score)
    return {
        "brier": summary["brier"],
        "intercepto": summary["intercepto"],
        "slope": summary["slope"],
        "ece_quantis": summary["ece_quantis"],
    }
