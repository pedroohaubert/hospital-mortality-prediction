"""Validação cruzada estratificada e repetida, com os mesmos folds para todos."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from desfecho.config import TrainConfig
from desfecho.metrics import (
    binary_rates,
    bootstrap_auc_ci,
    calibration_summary,
    delong_auc,
    delong_comparison,
    mean_score,
    safe_auc,
)
from desfecho.models import fresh_catboost, make_logistic
from desfecho.schema import AGE_COL, ID_COL, TARGET_COL
from desfecho.threshold import (
    select_threshold,
    threshold_at_sensitivity,
    youden_and_max_f1,
)

MODEL_KEYS = ("idade", "logistica", "catboost")
LogFn = Callable[[str], None]


def _noop(message: str) -> None:
    del message


def official_splits(
    n_rows: int,
    y: np.ndarray,
    cfg: TrainConfig | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Os mesmos 5×5 do treino entregue. A semente mora em TrainConfig."""
    cfg = cfg or TrainConfig()
    y = np.asarray(y)
    splitter = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.seed,
    )
    return list(splitter.split(np.zeros(n_rows), y))


def run_training(df: pd.DataFrame, cfg: TrainConfig, log: LogFn = _noop) -> dict:
    started = time.perf_counter()
    y = df[TARGET_COL].to_numpy(dtype=int)
    n = len(df)
    splits = official_splits(n, y, cfg)
    oof = {key: np.full((cfg.n_repeats, n), np.nan) for key in MODEL_KEYS}
    models = {key: [] for key in MODEL_KEYS}
    fold_rows = []
    cat_trees = []

    for fold_id, (train_idx, valid_idx) in enumerate(splits):
        repeat, fold = divmod(fold_id, cfg.n_splits)
        fold_seed = cfg.seed + 10007 * repeat + 17 * fold
        x_train = df.iloc[train_idx]
        x_valid = df.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        age_model = make_logistic(cfg, mode="idade")
        age_model.fit(x_train, y_train)
        p_age = age_model.predict_proba(x_valid)[:, 1]

        logistic = make_logistic(cfg, mode="full")
        logistic.fit(x_train, y_train)
        p_logistic = logistic.predict_proba(x_valid)[:, 1]

        catboost = fresh_catboost(cfg, fold_seed)
        catboost.fit(x_train, y_train)
        p_cat = catboost.predict_proba(x_valid)[:, 1]
        cat_trees.append(int(catboost.best_iteration_))

        oof["idade"][repeat, valid_idx] = p_age
        oof["logistica"][repeat, valid_idx] = p_logistic
        oof["catboost"][repeat, valid_idx] = p_cat
        models["idade"].append(age_model)
        models["logistica"].append(logistic)
        models["catboost"].append(catboost)

        fold_rows.append(
            {
                "repeticao": int(repeat),
                "fold": int(fold),
                "n_validacao": int(len(valid_idx)),
                "arvores_catboost": int(catboost.best_iteration_),
                "auc_idade": safe_auc(y_valid, p_age),
                "auc_logistica": safe_auc(y_valid, p_logistic),
                "auc_catboost": safe_auc(y_valid, p_cat),
                "auc_ensemble": safe_auc(y_valid, 0.5 * (p_logistic + p_cat)),
            }
        )
        log(
            f"repetição {repeat + 1}/{cfg.n_repeats} fold {fold + 1}/{cfg.n_splits} "
            f"AUC logística {fold_rows[-1]['auc_logistica']:.3f} "
            f"CatBoost {fold_rows[-1]['auc_catboost']:.3f} "
            f"árvores {catboost.best_iteration_}"
        )

    if np.isnan(np.concatenate([oof[key].ravel() for key in MODEL_KEYS])).any():
        raise RuntimeError("Ficou fold sem previsão. A validação não fechou.")

    mean_oof = {key: mean_score(oof[key]) for key in MODEL_KEYS}
    mean_oof["ensemble"] = cfg.weight_logistica * mean_oof["logistica"] + cfg.weight_catboost * mean_oof["catboost"]
    repeat_auc = _repeat_aucs(y, oof, cfg)

    aucs = {key: safe_auc(y, score) for key, score in mean_oof.items()}
    shipped, ship_reason = _choose_shipped(aucs, cfg)
    shipped_score = mean_oof[shipped]
    log(f"modelo que vai para a submissão: {shipped}. {ship_reason}")

    choice, curve = select_threshold(
        y,
        shipped_score,
        n_boot=cfg.threshold_boot,
        seed=cfg.seed,
        min_neg_rate=cfg.min_neg_rate,
        f1_margin=cfg.f1_margin,
    )
    nested = _nested_threshold(y, oof, shipped, cfg)
    log(
        f"limiar {choice.threshold:.6f} ({choice.level}) "
        f"LR− {choice.point['lr_neg']:.4f} F1 {choice.point['f1']:.4f}"
    )

    metrics = _metrics_block(
        y=y,
        mean_oof=mean_oof,
        repeat_auc=repeat_auc,
        fold_rows=fold_rows,
        cfg=cfg,
        shipped=shipped,
        ship_reason=ship_reason,
        choice=choice,
        curve=curve,
        nested=nested,
        cat_trees=cat_trees,
        elapsed=time.perf_counter() - started,
    )
    explanation = _explanation(models, df, cfg)
    oof_frame = _oof_frame(df, y, mean_oof)
    bundle = {
        "version": 1,
        "shipped": shipped,
        "ship_reason": ship_reason,
        "threshold": choice.threshold,
        "threshold_rule": choice.rule,
        "threshold_level": choice.level,
        "weights": {"logistica": cfg.weight_logistica, "catboost": cfg.weight_catboost},
        "config": cfg.to_dict(),
        "models": models,
        "cat_trees": cat_trees,
    }
    return {
        "metrics": metrics,
        "oof": oof_frame,
        "bundle": bundle,
        "explanation": explanation,
        "curve": curve,
    }


def predict_proba(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Média dos modelos dos folds. O limiar não é reestimado aqui."""
    shipped = bundle["shipped"]
    weights = bundle["weights"]
    if shipped == "ensemble":
        p_logistic = _average_proba(bundle["models"]["logistica"], df)
        p_cat = _average_proba(bundle["models"]["catboost"], df)
        return weights["logistica"] * p_logistic + weights["catboost"] * p_cat
    if shipped not in bundle["models"]:
        raise KeyError(f"Modelo desconhecido no pacote: {shipped}")
    return _average_proba(bundle["models"][shipped], df)


def _average_proba(models: list, df: pd.DataFrame) -> np.ndarray:
    if not models:
        raise RuntimeError("O pacote não tem modelos ajustados.")
    stacked = [model.predict_proba(df)[:, 1] for model in models]
    return np.mean(np.vstack(stacked), axis=0)


def _repeat_aucs(y: np.ndarray, oof: dict[str, np.ndarray], cfg: TrainConfig) -> dict[str, list[float]]:
    out = {key: [] for key in list(MODEL_KEYS) + ["ensemble"]}
    for repeat in range(cfg.n_repeats):
        logistic = oof["logistica"][repeat]
        catboost = oof["catboost"][repeat]
        out["idade"].append(safe_auc(y, oof["idade"][repeat]))
        out["logistica"].append(safe_auc(y, logistic))
        out["catboost"].append(safe_auc(y, catboost))
        out["ensemble"].append(safe_auc(y, cfg.weight_logistica * logistic + cfg.weight_catboost * catboost))
    return out


def _choose_shipped(aucs: dict[str, float], cfg: TrainConfig) -> tuple[str, str]:
    ensemble = aucs["ensemble"]
    age = aucs["idade"]
    logistic = aucs["logistica"]
    catboost = aucs["catboost"]
    if ensemble + cfg.min_auc_gain_over_age < age:
        return (
            "idade",
            "A média out-of-fold do ensemble ficou mais de 0,005 abaixo da logística só com a idade. "
            "Levo o modelo menor.",
        )
    gap = abs(logistic - catboost)
    if gap > cfg.max_auc_gap_to_keep_both:
        winner = "logistica" if logistic >= catboost else "catboost"
        return (
            winner,
            f"A diferença de AUC out-of-fold entre logística e CatBoost foi {gap:.3f}, acima de 0,02. "
            f"O ensemble médio levaria o pior dos dois. Fica só {winner}.",
        )
    return (
        "ensemble",
        "Logística e CatBoost ficaram a até 0,02 de AUC. A submissão usa a média simples "
        "das probabilidades, com peso 0,5, sem reajustar peso na mesma validação.",
    )


def _nested_threshold(y: np.ndarray, oof: dict[str, np.ndarray], shipped: str, cfg: TrainConfig) -> list[dict]:
    """Outra repetição, os mesmos pacientes.

    Cada repetição cobre todo mundo. O corte usa o rótulo de quem depois é medido.
    Não é validação aninhada nem disjunta. O nome antigo ficou no JSON entregue.
    """
    rows = []
    n_repeats = cfg.n_repeats
    if n_repeats < 2:
        return rows
    for held in range(n_repeats):
        selector = _score_from_repeats(oof, shipped, cfg, [r for r in range(n_repeats) if r != held])
        evaluation = _score_from_repeats(oof, shipped, cfg, [held])
        choice, _ = select_threshold(
            y,
            selector,
            n_boot=max(80, cfg.threshold_boot // 2),
            seed=cfg.seed + held + 1,
            min_neg_rate=cfg.min_neg_rate,
            f1_margin=cfg.f1_margin,
        )
        point = binary_rates(y, evaluation, choice.threshold)
        point["repeticao"] = held
        point["limiar"] = choice.threshold
        point["nivel"] = choice.level
        rows.append(point)
    return rows


def _score_from_repeats(oof: dict[str, np.ndarray], shipped: str, cfg: TrainConfig, repeats: list[int]) -> np.ndarray:
    if shipped == "ensemble":
        logistic = np.mean(oof["logistica"][repeats], axis=0)
        catboost = np.mean(oof["catboost"][repeats], axis=0)
        return cfg.weight_logistica * logistic + cfg.weight_catboost * catboost
    return np.mean(oof[shipped][repeats], axis=0)


def _metrics_block(
    *,
    y,
    mean_oof,
    repeat_auc,
    fold_rows,
    cfg: TrainConfig,
    shipped,
    ship_reason,
    choice,
    curve,
    nested,
    cat_trees,
    elapsed,
) -> dict:
    models = {}
    for key, score in mean_oof.items():
        boot = bootstrap_auc_ci(y, score, n_boot=cfg.auc_boot, seed=cfg.seed)
        delong = delong_auc(y, score)
        calib = calibration_summary(y, score)
        repeats = repeat_auc[key]
        models[key] = {
            "auc_media_oof": boot["auc"],
            "auc_bootstrap_ic95": [boot["ic95_inf"], boot["ic95_sup"]],
            "auc_delong_ic95": [delong["ic95_inf"], delong["ic95_sup"]],
            "auc_delong": delong["auc"],
            "auc_repeticoes": repeats,
            "auc_repeticoes_media": float(np.mean(repeats)),
            "auc_repeticoes_dp": float(np.std(repeats, ddof=1)) if len(repeats) > 1 else 0.0,
            "brier": calib["brier"],
            "calibracao_intercepto": calib["intercepto"],
            "calibracao_slope": calib["slope"],
            "ece_quantis": calib["ece_quantis"],
            "bins_calibracao": calib["bins"],
            "roc_fpr": calib["roc_fpr"],
            "roc_tpr": calib["roc_tpr"],
        }

    comparisons = []
    pairs = [
        ("ensemble", "logistica"),
        ("ensemble", "catboost"),
        ("ensemble", "idade"),
        ("logistica", "catboost"),
        ("logistica", "idade"),
        ("catboost", "idade"),
    ]
    for left, right in pairs:
        stat = delong_comparison(y, mean_oof[left], mean_oof[right])
        comparisons.append({"a": left, "b": right, **stat})

    alternatives = youden_and_max_f1(curve)
    sens_targets = {
        f"sens_{target:.2f}": threshold_at_sensitivity(y, mean_oof[shipped], target)
        for target in (0.90, 0.95, 0.98, 0.99)
    }
    nested_summary = _summarize_nested(nested)
    return {
        "modelo_submissao": shipped,
        "motivo_submissao": ship_reason,
        "limiar": choice.threshold,
        "limiar_regra": choice.rule,
        "limiar_nivel": choice.level,
        "limiar_p75_lr_neg": choice.p75_lr_neg,
        "limiar_media_boot_lr_neg": choice.mean_lr_neg_boot,
        "ponto_oof": choice.point,
        "ponto_oof_aviso": (
            "Limiar escolhido nestes mesmos escores out-of-fold. "
            "O ponto daqui é otimista em relação a um conjunto novo."
        ),
        "ponto_aninhado_por_repeticao": nested,
        "ponto_aninhado_media": nested_summary,
        "alternativas": {
            "youden": alternatives["youden"],
            "max_f1": alternatives["max_f1"],
            **sens_targets,
        },
        "modelos": models,
        "comparacoes_delong": comparisons,
        "folds": fold_rows,
        "catboost_arvores": cat_trees,
        "catboost_arvores_media": float(np.mean(cat_trees)) if cat_trees else None,
        "config": cfg.to_dict(),
        "n": int(len(y)),
        "prevalencia": float(np.mean(y)),
        "segundos": float(elapsed),
        "curva_limiar": [
            {
                "limiar": row["limiar"],
                "sensibilidade": row["sensibilidade"],
                "especificidade": row["especificidade"],
                "lr_neg": row["lr_neg"] if np.isfinite(row["lr_neg"]) else None,
                "f1": row["f1"],
                "taxa_prevista_0": row["taxa_prevista_0"],
            }
            for row in curve
        ],
    }


def _summarize_nested(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    keys = ["sensibilidade", "especificidade", "lr_neg", "f1", "taxa_prevista_0", "limiar"]
    summary = {"n_repeticoes": len(rows)}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=float)
        summary[f"{key}_media"] = float(np.mean(values))
        summary[f"{key}_min"] = float(np.min(values))
        summary[f"{key}_max"] = float(np.max(values))
    return summary


def _oof_frame(df: pd.DataFrame, y: np.ndarray, mean_oof: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": df[ID_COL].astype(str).to_numpy(),
            "desfecho": y.astype(int),
            "p_idade": mean_oof["idade"],
            "p_logistica": mean_oof["logistica"],
            "p_catboost": mean_oof["catboost"],
            "p_ensemble": mean_oof["ensemble"],
        }
    )


def _explanation(models: dict, df: pd.DataFrame, cfg: TrainConfig) -> dict:
    importances = []
    for model in models["catboost"]:
        importances.append(model.feature_importance())
    names = sorted(importances[0])
    mean_importance = []
    for name in names:
        values = [item[name] for item in importances]
        mean_importance.append(
            {"feature": name, "importancia": float(np.mean(values)), "dp": float(np.std(values, ddof=1))}
        )
    mean_importance.sort(key=lambda item: item["importancia"], reverse=True)

    coefs, references = _mean_binary_coefficients(models["logistica"])
    ages = [float(a) for a in np.linspace(0, 100, 21)]
    curve = _age_curve(models, df, ages, cfg)
    return {
        "importancia_catboost": mean_importance,
        "coeficientes_binarios_logistica": coefs,
        "referencias_categoricas": references,
        "curva_idade": curve,
    }


def _mean_binary_coefficients(models: list) -> tuple[list[dict], dict]:
    collected: dict[str, list[float]] = {}
    references: dict[str, list[str]] = {}
    for model in models:
        design = model.named_steps["design"]
        for col, ref in design.references_.items():
            references.setdefault(col, [])
            if ref not in references[col]:
                references[col].append(ref)
        names = list(design.get_feature_names_out())
        coef = model.named_steps["clf"].coef_.ravel()
        for name, value in zip(names, coef, strict=True):
            if name.startswith("idade_spline_") or "[" in name:
                continue
            collected.setdefault(name, []).append(float(value))
    rows = []
    for name, values in collected.items():
        rows.append(
            {
                "feature": name,
                "coef_medio": float(np.mean(values)),
                "dp": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "folds": len(values),
            }
        )
    rows.sort(key=lambda item: abs(item["coef_medio"]), reverse=True)
    return rows, references


def _age_curve(models: dict, df: pd.DataFrame, ages: list[float], cfg: TrainConfig) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    take = min(400, len(df))
    idx = rng.choice(len(df), size=take, replace=False)
    base = df.iloc[idx]
    points = []
    families = {
        "idade": models["idade"],
        "logistica": models["logistica"],
        "catboost": models["catboost"],
    }
    # Um subconjunto fixo de folds basta para o desenho. A métrica não usa esta curva.
    keep = list(range(0, len(models["logistica"]), cfg.n_splits))[: cfg.n_repeats]
    for age in ages:
        block = base.copy()
        block[AGE_COL] = age
        probs = {}
        for name, family in families.items():
            chosen = [family[i] for i in keep]
            stacked = np.vstack([model.predict_proba(block)[:, 1] for model in chosen])
            probs[name] = float(stacked.mean())
        points.append(
            {
                "idade": float(age),
                "so_idade": probs["idade"],
                "logistica": probs["logistica"],
                "catboost": probs["catboost"],
                "ensemble": cfg.weight_logistica * probs["logistica"] + cfg.weight_catboost * probs["catboost"],
            }
        )
    return points
