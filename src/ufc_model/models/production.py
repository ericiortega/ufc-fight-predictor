"""The production model: logistic regression (scaled), uncalibrated.

Chosen in Phase 3 on 2013-2026 walk-forward results (a choice made using test
years; see README). Explanations are exact linear SHAP values:
coefficient x (scaled value - training mean of the scaled value), in log-odds,
symmetrized per bout so a positive value favors the first-named fighter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ufc_model.config import Config
from ufc_model.features.elo import expected_score  # noqa: F401  (re-exported for callers)
from ufc_model.models.calibration import make_calibrator
from ufc_model.models.data import design_matrix, eligible, symmetrize, year_of
from ufc_model.models.evaluate import metrics
from ufc_model.models.explain import base_feature
from ufc_model.models.train import fit_lr, tune_lr

log = logging.getLogger(__name__)


@dataclass
class ProductionModel:
    pipeline: Pipeline
    columns: list[str]
    params: dict
    data_cutoff: date  # latest fight date in the training data
    n_train_fights: int
    train_start_year: int
    scaled_train_mean: np.ndarray

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Symmetric per-bout win probability for the AB fighter (needs both orientations)."""
        x = design_matrix(rows)[self.columns]
        return symmetrize(rows, self.pipeline.predict_proba(x)[:, 1])

    def contributions(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Linear SHAP per bout (index fight_id), folded to source features."""
        x = design_matrix(rows)[self.columns]
        scaler, lr = self.pipeline[0], self.pipeline[-1]
        z = (scaler.transform(x) - self.scaled_train_mean) * lr.coef_[0]
        c = pd.DataFrame(
            z,
            columns=self.columns,
            index=pd.MultiIndex.from_frame(rows[["fight_id", "orientation"]]),
        )
        c = c.T.groupby(base_feature).sum().T
        return (c.xs("AB", level="orientation") - c.xs("BA", level="orientation")) / 2


def _fit(rows: pd.DataFrame, train_mask: pd.Series, cfg: Config) -> ProductionModel:
    """Tune C on the last inner_val_years of ``train_mask``, then refit on all of it."""
    year = year_of(rows)
    last = int(year[train_mask].max())
    cut = last + 1 - cfg.models.inner_val_years
    fit_m, val_m = train_mask & (year < cut), train_mask & (year >= cut)
    x, y = design_matrix(rows), rows["label"]
    params = tune_lr(
        x[fit_m],
        y[fit_m].astype(int),
        x[val_m],
        None,
        rows[val_m],
        cfg.models.optuna_trials_lr,
        cfg.project.seed,
    )
    pipe = fit_lr(x[train_mask], y[train_mask].astype(int), params["C"])
    return ProductionModel(
        pipeline=pipe,
        columns=list(x.columns),
        params=params,
        data_cutoff=pd.Timestamp(rows.loc[train_mask, "event_date"].max()).date(),
        n_train_fights=int(train_mask.sum()) // 2,
        train_start_year=cfg.features.train_start_year,
        scaled_train_mean=pipe[0].transform(x[train_mask]).mean(axis=0),
    )


def recent_holdout(rows: pd.DataFrame, cfg: Config, days: int) -> dict:
    """Honest check on the most recent cards: train only on fights before the last
    ``days`` days, score those fights, compare to Elo, and fit a temperature as a
    drift check (T near 1 means the probabilities are still well calibrated)."""
    ok = eligible(rows, cfg.features.train_start_year)
    dates = pd.to_datetime(rows["event_date"])
    split = dates[ok].max() - pd.Timedelta(days=days)
    model = _fit(rows, ok & (dates <= split), cfg)
    test = model.predict(rows[ok & (dates > split)])
    y = test["label"].astype(int).to_numpy()
    m = metrics(y, test["p"].to_numpy())
    elo = metrics(y, test["elo_prob"].to_numpy())
    t = make_calibrator("temperature").fit(test["p"].to_numpy(), y).t
    return {
        "holdout_start": str((split + pd.Timedelta(days=1)).date()),
        "holdout_end": str(dates[ok].max().date()),
        "model": {k: round(float(v), 4) for k, v in m.items()},
        "elo": {k: round(float(v), 4) for k, v in elo.items()},
        "temperature": round(t, 4),
        "temperature_drift_flag": abs(t - 1) > cfg.production.temperature_drift_tolerance,
    }


def train_production(rows: pd.DataFrame, cfg: Config) -> ProductionModel:
    """Fit on every eligible fight (from train_start_year to the data cutoff)."""
    return _fit(rows, eligible(rows, cfg.features.train_start_year), cfg)
