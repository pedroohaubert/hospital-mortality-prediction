"""Contraste pareado no paciente. Fold não é unidade amostral."""

from __future__ import annotations

import numpy as np

from desfecho.metrics import binary_rates, delong_comparison, safe_auc, stratified_indices
from desfecho.threshold import select_threshold
from tabpfn_exp.protocol import N_BOOT, PRIMARY_CANDIDATE, REFERENCE, SEED


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
) -> dict[str, str | bool]:
    """Adota nunca de forma automática. LR− e F1 não entram aqui.

    Bootstrap e DeLong são condicionais aos escores já gravados. Não cobrem
    o retreino nem a escolha do modelo.
    """
    base = {
        "limiar_entra_na_decisao": False,
        "intervalos": "condicionais_aos_escores_fixos_sem_variancia_de_treino",
        "contraste": f"{PRIMARY_CANDIDATE}_menos_{REFERENCE}",
    }
    if not measured:
        return {**base, "decisao": "inconclusive", "motivo": "nao_medido"}
    if not offline_ok:
        return {**base, "decisao": "reject", "motivo": "offline_ou_checksum"}
    if None in (bootstrap_low, bootstrap_high, delong_low, delong_high):
        return {**base, "decisao": "inconclusive", "motivo": "faltam_intervalos"}
    boot = _side(float(bootstrap_low), float(bootstrap_high))
    delong = _side(float(delong_low), float(delong_high))
    if boot != delong:
        return {**base, "decisao": "inconclusive", "motivo": "bootstrap_e_delong_discordam"}
    if boot == "cobre":
        return {**base, "decisao": "inconclusive", "motivo": "intervalo_cobre_zero"}
    if boot == "abaixo":
        return {**base, "decisao": "reject", "motivo": "intervalo_de_auc_abaixo_de_zero"}
    return {
        **base,
        "decisao": "inconclusive",
        "motivo": "auc_favoravel_limiar_apenas_exploratorio",
    }


def repeat_holdout_reuses_labels(n_patients: int) -> dict[str, int | bool]:
    """Segurar uma repetição não tira ninguém da escolha do corte.

    Cada repetição tem escore para todos os pacientes. O vetor de seleção tem
    o mesmo comprimento, e `select_threshold` recebe o vetor inteiro de rótulos.
    """
    if n_patients < 1:
        raise ValueError("Sem pacientes.")
    return {
        "n_avaliados": n_patients,
        "n_rotulos_reusados": n_patients,
        "disjunto": False,
    }


def exploratory_operating_points(
    y: np.ndarray,
    repeats: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    min_neg_rate: float,
    f1_margin: float,
    eval_groups: list[np.ndarray],
) -> dict:
    """LR− e F1 só como exploração. Nenhum destes cortes é independente.

    `por_repeticao` reusa o rótulo de todo mundo. `por_grupo` não reusa o
    rótulo do grupo, mas o escore dos outros pacientes veio de modelo que
    treinou nesse grupo. CV aninhada de verdade não cabe no orçamento.
    """
    y = np.asarray(y).astype(int)
    repeats = np.asarray(repeats, dtype=float)
    overlap = repeat_holdout_reuses_labels(len(y))
    by_repeat = []
    for held in range(repeats.shape[0]):
        selector = np.mean(np.delete(repeats, held, axis=0), axis=0)
        choice, _ = select_threshold(
            y,
            selector,
            n_boot=n_boot,
            seed=seed + held + 1,
            min_neg_rate=min_neg_rate,
            f1_margin=f1_margin,
        )
        point = binary_rates(y, repeats[held], choice.threshold)
        point["repeticao"] = held
        point["limiar"] = choice.threshold
        point["rotulos_disjuntos"] = False
        by_repeat.append(point)
    mean_score = repeats.mean(axis=0)
    full_choice, _ = select_threshold(
        y,
        mean_score,
        n_boot=n_boot,
        seed=seed,
        min_neg_rate=min_neg_rate,
        f1_margin=f1_margin,
    )
    full = binary_rates(y, mean_score, full_choice.threshold)
    full["limiar"] = full_choice.threshold
    full["rotulos_disjuntos"] = False
    by_group = []
    for group_id, group in enumerate(eval_groups):
        group = np.asarray(group, dtype=int)
        selection = np.ones(len(y), dtype=bool)
        selection[group] = False
        if selection.sum() == 0 or (~selection).sum() == 0:
            raise ValueError("Grupo de limiar vazio.")
        if len(np.unique(y[selection])) < 2 or len(np.unique(y[group])) < 2:
            continue
        choice, _ = select_threshold(
            y[selection],
            mean_score[selection],
            n_boot=n_boot,
            seed=seed + 100 + group_id,
            min_neg_rate=min_neg_rate,
            f1_margin=f1_margin,
        )
        point = binary_rates(y[group], mean_score[group], choice.threshold)
        point["grupo"] = group_id
        point["limiar"] = choice.threshold
        point["n_selecao"] = int(selection.sum())
        point["n_avaliados"] = int(len(group))
        point["rotulos_disjuntos"] = True
        point["modelo_treinou_no_grupo"] = True
        by_group.append(point)
    return {
        "papel": "exploratorio",
        "entra_na_decisao": False,
        "repeticao_reusa_rotulo": overlap,
        "por_repeticao": by_repeat,
        "lr_neg_media_repeticao": float(np.mean([row["lr_neg"] for row in by_repeat])),
        "f1_media_repeticao": float(np.mean([row["f1"] for row in by_repeat])),
        "corte_na_amostra_inteira": full,
        "por_grupo_rotulo_disjunto": by_group,
        "lr_neg_media_grupo": float(np.mean([row["lr_neg"] for row in by_group])) if by_group else None,
        "f1_media_grupo": float(np.mean([row["f1"] for row in by_group])) if by_group else None,
    }


def audit_threshold_patients(y: np.ndarray) -> dict[str, bool | int]:
    """Mostra quem entra na escolha do corte e quem treinou o escore."""
    from desfecho.config import TrainConfig
    from desfecho.cv import official_splits

    y = np.asarray(y)
    cfg = TrainConfig()
    splits = official_splits(len(y), y, cfg)
    _, eval_idx = splits[0]
    eval_set = {int(i) for i in eval_idx}
    selection = set(range(len(y))) - eval_set
    saw_eval = False
    for fold in range(cfg.n_splits):
        train_idx, valid_idx = splits[fold]
        if {int(i) for i in valid_idx} & eval_set:
            continue
        if eval_set & {int(i) for i in train_idx}:
            saw_eval = True
            break
    return {
        "repeat_holdout_disjoint": False,
        "repeat_holdout_eval_labels_also_used": len(eval_set),
        "group_holdout_disjoint": eval_set.isdisjoint(selection),
        "group_models_trained_on_eval": saw_eval,
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
