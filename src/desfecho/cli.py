"""Linha de comando. No dia da prova, `predict` não baixa nada e não reescolhe o limiar."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from desfecho import __version__
from desfecho.config import TrainConfig
from desfecho.cv import predict_proba, run_training
from desfecho.data import dataset_checks, load_table
from desfecho.packaging import build_zip
from desfecho.schema import ID_COL, TARGET_COL
from desfecho.store import load_bundle, load_explanation, load_metrics, load_oof, save_training
from desfecho.submission import build_submission, validate_submission, write_submission
from desfecho.viz import save_age_rate, save_figures

app = typer.Typer(
    help="Treina, avalia, prevê e desenha o modelo de óbito hospitalar da ERAMIA-RS 2026.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _config(
    seed: int,
    splits: int,
    repeats: int,
    iterations: int,
    threshold_boot: int,
    auc_boot: int,
) -> TrainConfig:
    base = TrainConfig()
    return TrainConfig(
        seed=seed,
        n_splits=splits,
        n_repeats=repeats,
        cat_iterations=iterations,
        cat_learning_rate=base.cat_learning_rate,
        cat_depth=base.cat_depth,
        cat_l2_leaf_reg=base.cat_l2_leaf_reg,
        cat_early_stopping_rounds=base.cat_early_stopping_rounds,
        cat_inner_valid_size=base.cat_inner_valid_size,
        cat_one_hot_max_size=base.cat_one_hot_max_size,
        cat_thread_count=base.cat_thread_count,
        lr_C=base.lr_C,
        spline_knots=base.spline_knots,
        spline_degree=base.spline_degree,
        weight_logistica=base.weight_logistica,
        weight_catboost=base.weight_catboost,
        max_auc_gap_to_keep_both=base.max_auc_gap_to_keep_both,
        min_auc_gain_over_age=base.min_auc_gain_over_age,
        threshold_boot=threshold_boot,
        auc_boot=auc_boot,
        min_neg_rate=base.min_neg_rate,
        f1_margin=base.f1_margin,
    )


@app.callback()
def main() -> None:
    """Comandos do desafio. Use um subcomando."""


@app.command()
def train(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True, help="CSV de treino com DESFECHO."),
    out: Path = typer.Option(Path("artifacts"), "--out", "-o", help="Pasta dos artefatos."),
    seed: int = typer.Option(42, help="Semente da validação e do CatBoost."),
    splits: int = typer.Option(5, help="Folds estratificados."),
    repeats: int = typer.Option(5, help="Repetições da validação cruzada."),
    iterations: int = typer.Option(1000, help="Máximo de árvores do CatBoost, antes da parada."),
    threshold_boot: int = typer.Option(400, help="Reamostras para o p75 da LR−."),
    auc_boot: int = typer.Option(2000, help="Reamostras do intervalo da AUC."),
) -> None:
    """Ajusta os folds, escolhe o limiar e grava o modelo."""
    cfg = _config(seed, splits, repeats, iterations, threshold_boot, auc_boot)
    frame = load_table(data, require_target=True)
    checks = dataset_checks(frame)
    id_auc = checks["auc_id_numerico"]
    id_text = "n/d" if id_auc is None else f"{id_auc:.3f}"
    console.print(
        f"[bold]Treino[/bold] n={checks['n']}  prevalência de óbito={checks['prevalencia_obito']:.3f}  "
        f"AUC do ID={id_text}"
    )
    if checks["binarias_inesperadas"]:
        console.print(f"[yellow]Binárias fora de 0/1:[/yellow] {checks['binarias_inesperadas']}")
    result = run_training(frame, cfg, log=console.print)
    save_training(out, result, checks)
    metrics = result["metrics"]
    point = metrics["ponto_oof"]
    console.print(
        f"Modelo [bold]{metrics['modelo_submissao']}[/bold]  "
        f"AUC OOF {metrics['modelos'][metrics['modelo_submissao']]['auc_media_oof']:.3f}  "
        f"limiar {metrics['limiar']:.6f}  "
        f"LR− {point['lr_neg']:.4f}  F1 {point['f1']:.4f}"
    )
    console.print(f"Arquivos em {out.resolve()}")


@app.command()
def evaluate(
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a", exists=True, file_okay=False),
) -> None:
    """Mostra a AUC, o intervalo, a calibração e o ponto do limiar."""
    metrics = load_metrics(artifacts)
    table = Table(title="AUC na média out-of-fold")
    table.add_column("Modelo")
    table.add_column("AUC", justify="right")
    table.add_column("IC bootstrap 95%", justify="right")
    table.add_column("IC DeLong 95%", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Slope", justify="right")
    names = {
        "idade": "Idade",
        "logistica": "Logística",
        "catboost": "CatBoost",
        "ensemble": "Ensemble",
    }
    for key, label in names.items():
        block = metrics["modelos"][key]
        boot = block["auc_bootstrap_ic95"]
        delong = block["auc_delong_ic95"]
        table.add_row(
            label,
            f"{block['auc_media_oof']:.3f}",
            f"{boot[0]:.3f}–{boot[1]:.3f}",
            f"{delong[0]:.3f}–{delong[1]:.3f}",
            f"{block['brier']:.3f}",
            f"{block['calibracao_slope']:.2f}",
        )
    console.print(table)
    point = metrics["ponto_oof"]
    console.print(
        f"Submissão: [bold]{metrics['modelo_submissao']}[/bold]\n{metrics['motivo_submissao']}\n"
        f"Limiar [bold]{metrics['limiar']:.6f}[/bold] ({metrics['limiar_nivel']}). "
        f"No OOF: sens {point['sensibilidade']:.3f}, espec {point['especificidade']:.3f}, "
        f"LR− {point['lr_neg']:.4f}, F1 {point['f1']:.4f}, "
        f"previstos 0 {point['taxa_prevista_0']:.3f}."
    )
    nested = metrics.get("ponto_aninhado_media")
    if nested:
        console.print(
            "Outra repetição, os mesmos pacientes (não é corte disjunto): "
            f"LR− média {nested['lr_neg_media']:.4f} "
            f"(mín {nested['lr_neg_min']:.4f}, máx {nested['lr_neg_max']:.4f}), "
            f"F1 média {nested['f1_media']:.4f}."
        )
    console.print(f"[dim]{metrics['ponto_oof_aviso']}[/dim]")
    console.print(f"Pacote desfecho {__version__}.")


@app.command()
def predict(
    data: Path = typer.Option(..., "--data", "-d", exists=True, readable=True, help="CSV no formato do teste, sem rótulo."),
    model: Path = typer.Option(Path("artifacts/modelo.joblib"), "--model", "-m", exists=True, dir_okay=False),
    out: Path = typer.Option(Path("submissao.csv"), "--out", "-o", help="CSV id,prob,desfecho."),
) -> None:
    """Gera a submissão com o limiar gravado no modelo. Não usa rede."""
    frame = load_table(data, require_target=False)
    if TARGET_COL in frame.columns and frame[TARGET_COL].notna().any():
        console.print("[yellow]A coluna DESFECHO foi ignorada. A previsão não usa rótulo.[/yellow]")
    bundle = load_bundle(model)
    threshold = float(bundle["threshold"])
    probabilities = predict_proba(bundle, frame)
    submission = build_submission(frame[ID_COL].tolist(), probabilities, threshold)
    validate_submission(submission, threshold, expected_ids=frame[ID_COL].tolist())
    write_submission(submission, out)
    share_zero = float((submission["desfecho"] == 0).mean())
    console.print(
        f"{len(submission)} linhas em {out.resolve()}  limiar {threshold:.6f}  "
        f"previstos 0: {share_zero:.3f}  modelo {bundle['shipped']}"
    )


@app.command()
def visualize(
    artifacts: Path = typer.Option(Path("artifacts"), "--artifacts", "-a", exists=True, file_okay=False),
    out: Path = typer.Option(Path("figs"), "--out", "-o"),
    data: Path | None = typer.Option(None, "--data", "-d", help="CSV de treino, só para a curva de óbito por idade."),
) -> None:
    """Desenha ROC, calibração, limiar e importâncias a partir dos artefatos."""
    metrics = load_metrics(artifacts)
    oof = load_oof(artifacts)
    explanation = load_explanation(artifacts)
    paths = save_figures(artifacts, metrics, oof, explanation, out)
    if data is not None:
        frame = load_table(data, require_target=True)
        paths.append(save_age_rate(frame, out / "idade_treino.png"))
    for path in paths:
        console.print(path.resolve())


@app.command()
def package(
    root: Path = typer.Option(Path("."), "--root", exists=True, file_okay=False),
    out: Path = typer.Option(Path("dist/eramia-rs-2026.zip"), "--out", "-o"),
) -> None:
    """Monta o zip pedido no regulamento, com o documento de ferramentas."""
    path = build_zip(root, out)
    console.print(f"Zip em {path.resolve()} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
