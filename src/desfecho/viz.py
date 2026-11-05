"""Figuras da validação. Os números vêm dos arquivos já gravados, sem reajustar modelo."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from desfecho.schema import AGE_COL, TARGET_COL

COLORS = {
    "idade": "#6b7280",
    "logistica": "#1d4e89",
    "catboost": "#c2410c",
    "ensemble": "#047857",
}
LABELS = {
    "idade": "Logística só com idade",
    "logistica": "Logística com spline",
    "catboost": "CatBoost",
    "ensemble": "Média logística + CatBoost",
}


def save_figures(directory: Path, metrics: dict, oof: pd.DataFrame, explanation: dict, out: Path) -> list[Path]:
    del directory
    if "n" in metrics and len(oof) != int(metrics["n"]):
        raise ValueError("oof.csv e metrics.json não têm o mesmo número de linhas.")
    out.mkdir(parents=True, exist_ok=True)
    written = [
        _roc(metrics, out / "roc.png"),
        _calibration(metrics, out / "calibracao.png"),
        _threshold(metrics, out / "limiar.png"),
    ]
    if explanation.get("importancia_catboost"):
        written.append(_importance(explanation, out / "importancia.png"))
    if explanation.get("curva_idade"):
        written.append(_age_effect(explanation, out / "idade_efeito.png"))
    return written


def save_age_rate(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    age = pd.to_numeric(df[AGE_COL], errors="coerce")
    y = df[TARGET_COL].astype(int)
    bins = list(range(0, 101, 5))
    cut = pd.cut(age, bins=bins, right=False)
    table = pd.DataFrame({"faixa": cut, "y": y}).dropna()
    grouped = table.groupby("faixa", observed=True)["y"].agg(["mean", "count"])
    centers = np.array([(interval.left + interval.right) / 2 for interval in grouped.index])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(centers, grouped["mean"], color="#9f1239", marker="o", lw=1.8)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Idade (faixas de 5 anos)")
    ax.set_ylabel("Proporção de óbito no treino")
    ax.set_title("Óbito por idade na base de treino")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _roc(metrics: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for key, color in COLORS.items():
        block = metrics["modelos"][key]
        ax.plot(block["roc_fpr"], block["roc_tpr"], color=color, lw=1.8, label=f"{LABELS[key]} ({block['auc_media_oof']:.3f})")
    ax.plot([0, 1], [0, 1], color="#d1d5db", lw=1)
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("Sensibilidade")
    ax.set_title("ROC na média out-of-fold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _calibration(metrics: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color="#d1d5db", lw=1)
    for key, color in COLORS.items():
        bins = metrics["modelos"][key]["bins_calibracao"]
        if not bins:
            continue
        ax.plot(
            [b["prob_media"] for b in bins],
            [b["obs_media"] for b in bins],
            color=color,
            marker="o",
            lw=1.6,
            label=LABELS[key],
        )
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Calibração por quantis, média out-of-fold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _threshold(metrics: dict, path: Path) -> Path:
    curve = metrics["curva_limiar"]
    x = [row["limiar"] for row in curve]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(x, [row["f1"] for row in curve], color="#1d4e89", label="F1")
    ax.plot(x, [row["sensibilidade"] for row in curve], color="#9f1239", label="Sensibilidade")
    ax.plot(x, [row["especificidade"] for row in curve], color="#047857", label="Especificidade")
    lr = [row["lr_neg"] if row["lr_neg"] is not None else np.nan for row in curve]
    ax2 = ax.twinx()
    ax2.plot(x, lr, color="#c2410c", ls="--", label="LR−")
    chosen = metrics["limiar"]
    ax.axvline(chosen, color="#111827", lw=1, label=f"Limiar {chosen:.3f}")
    ax.set_xlabel("Limiar")
    ax.set_ylabel("Sensibilidade, especificidade, F1")
    ax2.set_ylabel("LR−")
    ax.set_title("Limiar no escore out-of-fold do modelo submetido")
    ax.set_xlim(min(x), max(x))
    ax.set_ylim(0, 1)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _importance(explanation: dict, path: Path) -> Path:
    rows = explanation["importancia_catboost"][:12]
    rows = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh([row["feature"] for row in rows], [row["importancia"] for row in rows], color="#1d4e89")
    ax.set_xlabel("Importância média (PredictionValuesChange)")
    ax.set_title("CatBoost, média dos folds")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _age_effect(explanation: dict, path: Path) -> Path:
    rows = explanation["curva_idade"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = [row["idade"] for row in rows]
    series = (
        ("so_idade", "idade"),
        ("logistica", "logistica"),
        ("catboost", "catboost"),
        ("ensemble", "ensemble"),
    )
    for key, color_key in series:
        ax.plot(x, [row[key] for row in rows], color=COLORS[color_key], lw=1.8, label=LABELS[color_key])
    ax.set_xlabel("Idade colocada no lugar da idade original")
    ax.set_ylabel("Probabilidade média")
    ax.set_title("Efeito da idade, demais colunas como na amostra")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
