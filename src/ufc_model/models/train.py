"""Model fitting and Optuna tuning (logistic regression, LightGBM seed ensemble).

Tuning objective: symmetric log loss on the time-ordered inner validation block
(the last years of the training window), never the test year.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from ufc_model.models.data import symmetrize

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

LGBM_FIXED = {
    "objective": "binary",
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "bagging_freq": 1,
    "n_jobs": 4,
}


def fit_lr(x: pd.DataFrame, y: pd.Series, c: float) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000)).fit(x, y)


@dataclass
class LGBMEnsemble:
    """Average of LightGBM models that differ only in their random seed."""

    params: dict
    seeds: list[int]
    models: list[lgb.LGBMClassifier] = field(default_factory=list)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LGBMEnsemble:
        self.models = [
            lgb.LGBMClassifier(**LGBM_FIXED, **self.params, random_state=s).fit(x, y)
            for s in self.seeds
        ]
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(x)[:, 1] for m in self.models], axis=0)


def _sym_logloss(rows: pd.DataFrame, p: np.ndarray) -> float:
    s = symmetrize(rows, p)
    return log_loss(s["label"].astype(int), s["p"].clip(1e-6, 1 - 1e-6), labels=[0, 1])


def tune_lr(xf, yf, xv, yv, rows_v, n_trials: int, seed: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        c = trial.suggest_float("C", 1e-4, 10.0, log=True)
        return _sym_logloss(rows_v, fit_lr(xf, yf, c).predict_proba(xv)[:, 1])

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def tune_lgbm(xf, yf, xv, yv, rows_v, n_trials: int, seed: int) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 48),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }
        m = lgb.LGBMClassifier(**LGBM_FIXED, **params, random_state=seed).fit(xf, yf)
        return _sym_logloss(rows_v, m.predict_proba(xv)[:, 1])

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
