"""Logística com spline de idade e CatBoost. Nada aqui vê o fold de validação."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

from desfecho.config import TrainConfig
from desfecho.data import prepare_frame
from desfecho.schema import AGE_COL, BINARY_COLS, CAT_COLS


class LogisticDesign(BaseEstimator, TransformerMixin):
    """Spline na idade, binárias cruas e one-hot com a moda como referência.

    `mode="idade"` deixa só a spline. O ajuste da spline e das categorias
    acontece em `fit`, que o pipeline chama só com as linhas de treino do fold.
    """

    def __init__(self, mode: str = "full", n_knots: int = 5, degree: int = 3):
        self.mode = mode
        self.n_knots = n_knots
        self.degree = degree

    def fit(self, X, y=None):
        frame = prepare_frame(X)
        self.medians_ = {}
        numeric_cols = [AGE_COL] if self.mode == "idade" else [AGE_COL, *BINARY_COLS]
        for col in numeric_cols:
            values = frame[col].to_numpy(dtype=float)
            self.medians_[col] = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        age = self._impute_age(frame).reshape(-1, 1)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots="quantile",
            extrapolation="linear",
            include_bias=False,
        )
        self.spline_.fit(age)
        self.references_ = {}
        self.levels_ = {}
        if self.mode != "idade":
            for col in CAT_COLS:
                text = self._fill_cat(frame[col])
                mode = str(pd.Series(text).mode(dropna=True).iloc[0])
                rest = sorted(v for v in pd.unique(text) if v != mode)
                self.references_[col] = mode
                self.levels_[col] = [mode, *rest]
        self.feature_names_ = self._names()
        return self

    def transform(self, X):
        frame = prepare_frame(X)
        age = self._impute_age(frame).reshape(-1, 1)
        parts = [self.spline_.transform(age)]
        if self.mode != "idade":
            binary = np.column_stack(
                [frame[col].fillna(self.medians_[col]).to_numpy(dtype=float) for col in BINARY_COLS]
            )
            parts.append(binary)
            for col in CAT_COLS:
                parts.append(self._one_hot(col, frame[col]))
        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)

    def _impute_age(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[AGE_COL].fillna(self.medians_[AGE_COL]).to_numpy(dtype=float)

    @staticmethod
    def _fill_cat(series: pd.Series) -> np.ndarray:
        text = series.astype("object").where(series.notna(), "__AUSENTE__")
        return text.astype(str).to_numpy()

    def _one_hot(self, col: str, series: pd.Series) -> np.ndarray:
        known = set(self.levels_[col])
        reference = self.references_[col]
        text = self._fill_cat(series)
        collapsed = np.where(np.isin(text, list(known)), text, reference)
        dummies = [collapsed == level for level in self.levels_[col][1:]]
        if not dummies:
            return np.zeros((len(series), 0), dtype=float)
        return np.column_stack(dummies).astype(float)

    def _names(self) -> list[str]:
        n_spline = int(self.spline_.n_features_out_)
        names = [f"idade_spline_{i}" for i in range(n_spline)]
        if self.mode == "idade":
            return names
        names.extend(BINARY_COLS)
        for col in CAT_COLS:
            ref = self.references_[col]
            for level in self.levels_[col][1:]:
                names.append(f"{col}[{level}|ref={ref}]")
        return names


def make_logistic(cfg: TrainConfig, *, mode: str) -> Pipeline:
    return Pipeline(
        [
            ("design", LogisticDesign(mode=mode, n_knots=cfg.spline_knots, degree=cfg.spline_degree)),
            (
                "clf",
                LogisticRegression(
                    C=cfg.lr_C,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


class CatBoostFrameModel(BaseEstimator, ClassifierMixin):
    """CatBoost em data frame. A profundidade da parada vem de um recorte interno.

    O recorte interno é sorteado só dentro do treino do fold. O modelo que
    prevê a validação é reajustado nesse treino inteiro, com o número de
    árvores já escolhido.
    """

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        early_stopping_rounds: int = 50,
        inner_valid_size: float = 0.2,
        one_hot_max_size: int = 10,
        thread_count: int = 4,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.inner_valid_size = inner_valid_size
        self.one_hot_max_size = one_hot_max_size
        self.thread_count = thread_count

    def fit(self, X, y):
        frame = prepare_frame(X)
        target = np.asarray(y).astype(int)
        n_trees = self._choose_trees(frame, target)
        self.best_iteration_ = int(n_trees)
        self.model_ = self._booster(n_trees)
        self.model_.fit(frame, target, cat_features=CAT_COLS, verbose=False)
        self.classes_ = np.array([0, 1])
        self.feature_names_ = list(frame.columns)
        return self

    def predict_proba(self, X):
        frame = prepare_frame(X)
        positive = self.model_.predict_proba(frame)[:, 1]
        positive = np.asarray(positive, dtype=float)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importance(self) -> dict[str, float]:
        values = self.model_.get_feature_importance(type="PredictionValuesChange")
        return {name: float(value) for name, value in zip(self.feature_names_, values, strict=True)}

    def _choose_trees(self, frame: pd.DataFrame, target: np.ndarray) -> int:
        if self.early_stopping_rounds <= 0 or len(target) < 80 or len(np.unique(target)) < 2:
            return int(self.iterations)
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.inner_valid_size,
            random_state=self.random_seed,
        )
        fit_idx, valid_idx = next(splitter.split(frame, target))
        probe = self._booster(self.iterations)
        probe.fit(
            frame.iloc[fit_idx],
            target[fit_idx],
            cat_features=CAT_COLS,
            eval_set=(frame.iloc[valid_idx], target[valid_idx]),
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
        )
        best = probe.get_best_iteration()
        if best is None:
            trees = int(probe.tree_count_)
        else:
            trees = int(best) + 1
        return max(1, min(trees, int(self.iterations)))

    def _booster(self, iterations: int) -> CatBoostClassifier:
        return CatBoostClassifier(
            iterations=int(iterations),
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=self.random_seed,
            one_hot_max_size=self.one_hot_max_size,
            thread_count=self.thread_count,
            allow_writing_files=False,
            verbose=False,
        )


def fresh_catboost(cfg: TrainConfig, seed: int) -> CatBoostFrameModel:
    return CatBoostFrameModel(
        iterations=cfg.cat_iterations,
        learning_rate=cfg.cat_learning_rate,
        depth=cfg.cat_depth,
        l2_leaf_reg=cfg.cat_l2_leaf_reg,
        random_seed=seed,
        early_stopping_rounds=cfg.cat_early_stopping_rounds,
        inner_valid_size=cfg.cat_inner_valid_size,
        one_hot_max_size=cfg.cat_one_hot_max_size,
        thread_count=cfg.cat_thread_count,
    )


def clone_estimator(estimator):
    return clone(estimator)
