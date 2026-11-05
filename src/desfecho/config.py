"""Hiperparâmetros fixados antes da validação que será reportada.

Não há busca em grade na mesma CV do resultado. Trocar um número aqui muda
o objeto que a validação estima; o motivo de cada escolha está na nota de pesquisa.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 5
    cat_iterations: int = 1000
    cat_learning_rate: float = 0.05
    cat_depth: int = 6
    cat_l2_leaf_reg: float = 3.0
    cat_early_stopping_rounds: int = 50
    cat_inner_valid_size: float = 0.2
    cat_one_hot_max_size: int = 10
    cat_thread_count: int = 4
    lr_C: float = 1.0
    spline_knots: int = 5
    spline_degree: int = 3
    weight_logistica: float = 0.5
    weight_catboost: float = 0.5
    # Ensemble só sai se a logística e o CatBoost ficarem a até 0,02 de AUC.
    # Abaixo disso, o pior modelo não entra. Se o ensemble perder da idade
    # por mais de 0,005, vai a logística só com a idade.
    max_auc_gap_to_keep_both: float = 0.02
    min_auc_gain_over_age: float = 0.005
    threshold_boot: int = 400
    auc_boot: int = 2000
    min_neg_rate: float = 0.08
    f1_margin: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)
