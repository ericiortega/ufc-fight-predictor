"""Walk-forward evaluation: ``python -m ufc_model.models.evaluate``.

For each test year Y (expanding window, train strictly before Y):
  1. Tune hyperparameters with Optuna: fit on years < Y-2, score on Y-2..Y-1.
  2. Fit on years < Y-2 and predict Y-2..Y-1, then fit the calibrators on those
     time-ordered validation predictions.
  3. Refit on all years < Y with the tuned parameters and predict Y. Apply each
     calibrator.
Probabilities are symmetrized per fight. Nothing from Y is used before predicting Y.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ufc_model import db
from ufc_model.config import Config, get_config
from ufc_model.logging_setup import setup_logging
from ufc_model.models.calibration import make_calibrator
from ufc_model.models.data import design_matrix, symmetrize, walk_forward
from ufc_model.models.train import LGBMEnsemble, fit_lr, tune_lgbm, tune_lr

log = logging.getLogger("ufc_model.models.evaluate")

META = [
    "event_date",
    "fighter_name",
    "opponent_name",
    "label",
    "elo_prob",
    "meta_n_fights",
    "meta_opp_n_fights",
    "meta_profile_complete",
    "is_title",
    "five_rounds",
    "weight_class",
    "diff_win_rate",
    "diff_n_fights",
    "diff_age_years",
]


class LeakageAlarm(RuntimeError):
    """Accuracy above the configured threshold: treat as a leakage bug until proven otherwise."""


# ---------------------------------------------------------------- metrics


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Exact 50/50 predictions get half credit."""
    p, y = np.asarray(p), np.asarray(y)
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": len(y),
        "accuracy": accuracy(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
    }


# ---------------------------------------------------------------- walk-forward


@dataclass
class RunResult:
    preds: pd.DataFrame  # one row per (fight, model, calibration)
    params: dict[str, dict[int, dict]]  # model -> test year -> tuned params


def run_walk_forward(
    rows: pd.DataFrame,
    cfg: Config,
    start_year: int,
    models: tuple[str, ...] = ("lr", "lgbm"),
    drop: tuple[str, ...] = (),
    fixed_params: dict[str, dict[int, dict]] | None = None,
    tag: str = "",
) -> RunResult:
    mc, seed = cfg.models, cfg.project.seed
    x_all = design_matrix(rows, drop)
    y_all = rows["label"]
    out, params = [], {m: {} for m in models}
    for w in walk_forward(rows, start_year, tuple(mc.test_years), mc.inner_val_years):
        t0 = time.time()
        xf, yf = x_all[w.inner_fit], y_all[w.inner_fit].astype(int)
        xv, rv = x_all[w.inner_val], rows[w.inner_val]
        xt, yt = x_all[w.train], y_all[w.train].astype(int)
        xs, rs = x_all[w.test], rows[w.test]
        for model in models:
            if fixed_params is not None:
                best = fixed_params[model][w.test_year]
            elif model == "lr":
                best = tune_lr(xf, yf, xv, None, rv, mc.optuna_trials_lr, seed)
            else:
                best = tune_lgbm(xf, yf, xv, None, rv, mc.optuna_trials_lgbm, seed)
            params[model][w.test_year] = best
            if model == "lr":
                inner, full = fit_lr(xf, yf, best["C"]), fit_lr(xt, yt, best["C"])
                pv, ps = inner.predict_proba(xv)[:, 1], full.predict_proba(xs)[:, 1]
            else:
                pv = LGBMEnsemble(best, mc.lgbm_seeds).fit(xf, yf).predict_proba(xv)
                ps = LGBMEnsemble(best, mc.lgbm_seeds).fit(xt, yt).predict_proba(xs)
            val, test = symmetrize(rv, pv), symmetrize(rs, ps)
            for cal_name in mc.calibrations:
                cal = make_calibrator(cal_name).fit(val["p"].to_numpy(), val["label"].to_numpy())
                frame = test[META].copy()
                frame["p"] = cal.transform(test["p"].to_numpy())
                frame["model"], frame["calibration"] = model + tag, cal_name
                frame["test_year"] = w.test_year
                out.append(frame.reset_index())
        log.info(
            "%s%s start=%d test %d: %d train rows, %d test fights (%.0fs)",
            ",".join(models),
            tag,
            start_year,
            w.test_year,
            int(w.train.sum()),
            len(rs) // 2,
            time.time() - t0,
        )
    return RunResult(pd.concat(out, ignore_index=True), params)


def baseline_preds(rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Elo (probabilistic) and three simple pick-only baselines, on the same test fights."""
    mc = cfg.models
    frames = []
    for w in walk_forward(
        rows, cfg.features.train_start_year, tuple(mc.test_years), mc.inner_val_years
    ):
        ab = rows[w.test & (rows["orientation"] == "AB")].set_index("fight_id")
        for name, p in {
            "elo": ab["elo_prob"],
            "higher_win_rate": 0.5 + 0.5 * np.sign(ab["diff_win_rate"]),
            "more_ufc_fights": 0.5 + 0.5 * np.sign(ab["diff_n_fights"]),
            "younger": 0.5 - 0.5 * np.sign(ab["diff_age_years"]),
        }.items():
            f = ab[META].copy()
            f["p"], f["model"], f["calibration"], f["test_year"] = p, name, "none", w.test_year
            frames.append(f.reset_index())
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- reporting


SUBSETS = {
    "all": lambda d: pd.Series(True, index=d.index),
    "both_3plus_prior_fights": lambda d: (d["meta_n_fights"] >= 3) & (d["meta_opp_n_fights"] >= 3),
    "any_0to2_prior_fights": lambda d: (d["meta_n_fights"] < 3) | (d["meta_opp_n_fights"] < 3),
    "profiles_complete": lambda d: d["meta_profile_complete"].astype(bool),
    "any_profile_imputed": lambda d: ~d["meta_profile_complete"].astype(bool),
    "title_or_five_round": lambda d: (d["is_title"] == 1) | (d["five_rounds"] == 1),
}
PICK_ONLY = {"higher_win_rate", "more_ufc_fights", "younger"}


def metric_table(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby(by, sort=True):
        m = metrics(g["label"].astype(int).to_numpy(), g["p"].to_numpy())
        if g["model"].iloc[0] in PICK_ONLY:
            m["log_loss"] = m["brier"] = np.nan  # hard picks: accuracy only
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)), **m})
    return pd.DataFrame(rows)


def subset_table(preds: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name, fn in SUBSETS.items():
        sub = preds[fn(preds)]
        if not sub.empty:
            parts.append(metric_table(sub, ["model", "calibration"]).assign(subset=name))
    return pd.concat(parts, ignore_index=True)


def check_leakage_alarm(preds: pd.DataFrame, threshold: float) -> None:
    pooled = metric_table(preds, ["model", "calibration"])
    yearly = metric_table(preds, ["model", "calibration", "test_year"])
    hits = pd.concat([pooled.assign(test_year="pooled"), yearly])
    hits = hits[hits["accuracy"] > threshold]
    if not hits.empty:
        raise LeakageAlarm(
            f"Accuracy above {threshold:.0%}; stopping to investigate:\n"
            f"{hits.to_string(index=False)}"
        )


def main() -> None:
    setup_logging()
    cfg = get_config()
    mc = cfg.models
    out_dir = mc.reports_dir
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    con = db.connect(cfg.paths.db_path)
    rows = con.sql("select * from fight_features order by event_date, fight_id, orientation").df()
    con.close()
    rows = rows.reset_index(drop=True)
    alarm = mc.leakage_accuracy_alarm

    base = baseline_preds(rows, cfg)
    main_run = run_walk_forward(rows, cfg, cfg.features.train_start_year)
    preds = pd.concat([base, main_run.preds], ignore_index=True)
    check_leakage_alarm(preds, alarm)  # stop before any further experiment

    profile_run = run_walk_forward(
        rows,
        cfg,
        cfg.features.train_start_year,
        models=("lgbm",),
        drop=tuple(mc.profile_features),
        fixed_params={"lgbm": main_run.params["lgbm"]},
        tag="_no_profile",
    )
    check_leakage_alarm(profile_run.preds, alarm)
    start_runs = {}
    for start in mc.start_year_experiment:
        if start == cfg.features.train_start_year:
            start_runs[start] = main_run.preds
            continue
        r = run_walk_forward(rows, cfg, start, tag=f"_start{start}")
        check_leakage_alarm(r.preds, alarm)
        start_runs[start] = r.preds

    all_preds = pd.concat(
        [
            preds,
            profile_run.preds,
            *[p for s, p in start_runs.items() if s != cfg.features.train_start_year],
        ],
        ignore_index=True,
    )
    all_preds.to_parquet(out_dir / "predictions.parquet", index=False)  # ~3 MB vs 30 MB csv
    metric_table(all_preds, ["model", "calibration"]).to_csv(
        out_dir / "metrics_pooled.csv", index=False
    )
    metric_table(all_preds, ["model", "calibration", "test_year"]).to_csv(
        out_dir / "metrics_by_year.csv", index=False
    )
    subset_table(all_preds).to_csv(out_dir / "metrics_subsets.csv", index=False)
    (out_dir / "tuned_params.json").write_text(
        json.dumps(
            {m: {str(y): p for y, p in v.items()} for m, v in main_run.params.items()}, indent=2
        )
    )

    from ufc_model.models import plots  # matplotlib only when reporting

    plots.make_all(all_preds, out_dir / "figures")
    log.info("Reports written to %s", out_dir)


if __name__ == "__main__":
    main()
