"""Profile-missingness leakage diagnostics: ``python -m ufc_model.features.diagnostics``.

Profiles are scraped today, so a missing height/reach/stance/DOB partly reveals a
short (often losing) career. All checks use walk-forward splits (train on years
before Y, test on year Y) over the training rows.

1. ``raw_missingness_test``: does the *raw* missingness pattern carry outcome
   signal beyond the number of prior UFC fights? (M0: n_prior only; M1: + flags.)
   This measures the leak present in the source data.

2. ``fed_feature_leak_test``: do the profile features we actually feed the model
   (imputed height/reach/age/stance diffs) let a model *recover* that signal?
   - identifiability: AUC for predicting a fighter's own missingness from
     n_prior + fed profile features, vs n_prior alone
   - capture: share of the raw flags' outcome gain that disappears once fed
     profile features are present (if fed features encoded missingness, adding
     the flags on top would add little)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.features.build import PROFILE_FIELDS
from ufc_model.logging_setup import setup_logging

log = logging.getLogger(__name__)

BASE = ["log_n_self", "log_n_opp", "log_n_diff"]
OWN_FLAGS = [f"meta_missing_{f}" for f in PROFILE_FIELDS]
FLAGS = OWN_FLAGS + [f"meta_opp_missing_{f}" for f in PROFILE_FIELDS]
FED_PROFILE = [
    "diff_height_in",
    "diff_reach_in",
    "diff_age_years",
    "diff_stance_orthodox",
    "diff_stance_southpaw",
    "diff_stance_switch",
]


def _design(rows: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=rows.index)
    x["log_n_self"] = np.log1p(rows["meta_n_fights"])
    x["log_n_opp"] = np.log1p(rows["meta_opp_n_fights"])
    x["log_n_diff"] = x["log_n_self"] - x["log_n_opp"]
    for c in FLAGS:
        x[c] = rows[c].astype(float)
    for c in FED_PROFILE:
        x[c] = rows[c]
    return x


def make_logistic(seed: int):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def make_gbm(seed: int):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed
    )


def walk_forward(
    rows: pd.DataFrame,
    x: pd.DataFrame,
    target: pd.Series,
    cols: list[str],
    make_model: Callable,
    years: tuple[int, int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled out-of-sample predictions: train on years < Y, test on year Y (AB rows)."""
    year = pd.to_datetime(rows["event_date"]).dt.year
    ys, ps = [], []
    for y in range(years[0], years[1] + 1):
        tr, te = year < y, (year == y) & (rows["orientation"] == "AB")
        if te.sum() == 0:
            continue
        model = make_model(seed).fit(x.loc[tr, cols], target[tr])
        ps.append(model.predict_proba(x.loc[te, cols])[:, 1])
        ys.append(target[te].to_numpy())
    return np.concatenate(ys), np.concatenate(ps)


@dataclass(frozen=True)
class RawMissingnessResult:
    model: str
    n_test_fights: int
    logloss_base: float
    logloss_with_flags: float
    acc_base: float
    acc_with_flags: float

    @property
    def logloss_gain(self) -> float:
        return self.logloss_base - self.logloss_with_flags

    @property
    def accuracy_gain(self) -> float:
        return self.acc_with_flags - self.acc_base


@dataclass(frozen=True)
class FedLeakResult:
    auc_missing_base: float
    auc_missing_fed: float
    raw_flag_gain: float  # logloss gain of flags over n_prior
    residual_flag_gain: float  # logloss gain of flags over n_prior + fed profile features

    @property
    def identifiability_gain(self) -> float:
        return self.auc_missing_fed - self.auc_missing_base

    @property
    def captured_share(self) -> float:
        """Share of the raw missingness signal already present in fed features."""
        return 1 - self.residual_flag_gain / self.raw_flag_gain if self.raw_flag_gain > 0 else 0.0


def _training_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    rows = rows[rows["in_training"]].copy()
    return rows, _design(rows), rows["label"].astype(int)


def raw_missingness_test(
    rows: pd.DataFrame, years: tuple[int, int], seed: int
) -> list[RawMissingnessResult]:
    rows, x, y = _training_rows(rows)
    out = []
    for name, make in (("logistic", make_logistic), ("gbm", make_gbm)):
        yt, pb = walk_forward(rows, x, y, BASE, make, years, seed)
        _, pf = walk_forward(rows, x, y, BASE + FLAGS, make, years, seed)
        out.append(
            RawMissingnessResult(
                name,
                len(yt),
                log_loss(yt, pb),
                log_loss(yt, pf),
                accuracy_score(yt, pb > 0.5),
                accuracy_score(yt, pf > 0.5),
            )
        )
    return out


def fed_feature_leak_test(rows: pd.DataFrame, years: tuple[int, int], seed: int) -> FedLeakResult:
    rows, x, y = _training_rows(rows)
    missing = rows[OWN_FLAGS].any(axis=1).astype(int)
    auc = {}
    for key, cols in (("base", BASE), ("fed", BASE + FED_PROFILE)):
        mt, mp = walk_forward(rows, x, missing, cols, make_gbm, years, seed)
        auc[key] = roc_auc_score(mt, mp)

    def ll(cols: list[str]) -> float:
        yt, p = walk_forward(rows, x, y, cols, make_gbm, years, seed)
        return log_loss(yt, p)

    return FedLeakResult(
        auc["base"],
        auc["fed"],
        raw_flag_gain=ll(BASE) - ll(BASE + FLAGS),
        residual_flag_gain=ll(BASE + FED_PROFILE) - ll(BASE + FED_PROFILE + FLAGS),
    )


def main() -> None:
    setup_logging()
    cfg = get_config()
    con = db.connect(cfg.paths.db_path)
    rows = con.sql("select * from fight_features").df()
    con.close()
    years, seed = tuple(cfg.leakage.test_years), cfg.project.seed
    print("1) Raw missingness flags vs n_prior_fights alone (outcome)")
    for r in raw_missingness_test(rows, years, seed):
        print(
            f"   {r.model:9s} n={r.n_test_fights}  logloss {r.logloss_base:.4f} -> "
            f"{r.logloss_with_flags:.4f} (gain {r.logloss_gain:+.4f})  acc {r.acc_base:.3f} -> "
            f"{r.acc_with_flags:.3f} (gain {r.accuracy_gain:+.3f})"
        )
    f = fed_feature_leak_test(rows, years, seed)
    print("2) Do fed (imputed) profile features carry it?")
    print(
        f"   AUC(own missingness): n_prior {f.auc_missing_base:.3f} -> + fed profile "
        f"{f.auc_missing_fed:.3f} (gain {f.identifiability_gain:+.3f})"
    )
    print(
        f"   flag outcome gain: over n_prior {f.raw_flag_gain:+.4f}, over n_prior + fed profile "
        f"{f.residual_flag_gain:+.4f} -> fed features capture {f.captured_share:.0%}"
    )


if __name__ == "__main__":
    main()
