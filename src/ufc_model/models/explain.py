"""SHAP explanations: ``python -m ufc_model.models.explain [--event "UFC 334"]``.

Fits the current model on every eligible fight (same recipe as each walk-forward
step: Optuna and calibration on the last two years, then refit on everything),
then writes:
- global importance: mean |SHAP| over recent fights, one-hot columns folded back
- per-bout predictions and top drivers for an upcoming card

SHAP values are in log-odds for the LightGBM ensemble (averaged over seeds),
symmetrized per bout as (SHAP_AB - SHAP_BA) / 2, so a positive value favors the
first-named fighter. Calibration rescales probabilities but not the ranking of
drivers.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import shap

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.logging_setup import setup_logging
from ufc_model.models import plots
from ufc_model.models.calibration import make_calibrator
from ufc_model.models.data import design_matrix, eligible, symmetrize, year_of
from ufc_model.models.train import LGBMEnsemble, tune_lgbm

log = logging.getLogger("ufc_model.models.explain")


def base_feature(col: str) -> str:
    for cat in ("weight_class_", "stance_matchup_"):
        if col.startswith(cat):
            return cat.rstrip("_")
    return col.removeprefix("diff_")


def fold(values: pd.DataFrame) -> pd.DataFrame:
    """Sum one-hot SHAP columns back into their source feature."""
    return values.T.groupby(base_feature).sum().T


def ensemble_shap(ens: LGBMEnsemble, x: pd.DataFrame) -> pd.DataFrame:
    vals = []
    for m in ens.models:
        sv = shap.TreeExplainer(m.booster_).shap_values(x)
        vals.append(sv[1] if isinstance(sv, list) else sv)
    return pd.DataFrame(np.mean(vals, axis=0), index=x.index, columns=x.columns)


def fit_current(rows: pd.DataFrame, cfg, calibration: str):
    mc, seed = cfg.models, cfg.project.seed
    ok = eligible(rows, cfg.features.train_start_year)
    year = year_of(rows)
    last = int(year[ok].max())
    cut = last + 1 - mc.inner_val_years
    fit_m, val_m = ok & (year < cut), ok & (year >= cut)
    x = design_matrix(rows)
    y = rows["label"]
    params = tune_lgbm(
        x[fit_m], y[fit_m].astype(int), x[val_m], None, rows[val_m], mc.optuna_trials_lgbm, seed
    )
    pv = (
        LGBMEnsemble(params, mc.lgbm_seeds)
        .fit(x[fit_m], y[fit_m].astype(int))
        .predict_proba(x[val_m])
    )
    val = symmetrize(rows[val_m], pv)
    cal = make_calibrator(calibration).fit(val["p"].to_numpy(), val["label"].to_numpy())
    ens = LGBMEnsemble(params, mc.lgbm_seeds).fit(x[ok], y[ok].astype(int))
    return ens, cal, params, x.columns, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="UFC 334")
    ap.add_argument("--calibration", default="none")
    args = ap.parse_args()
    setup_logging()
    cfg = get_config()
    out = cfg.models.reports_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    con = db.connect(cfg.paths.db_path)
    rows = con.sql("select * from fight_features order by event_date, fight_id, orientation").df()
    upcoming = con.sql("select * from upcoming_features").df()
    con.close()
    rows = rows.reset_index(drop=True)

    ens, cal, params, cols, ok = fit_current(rows, cfg, args.calibration)
    log.info("Current model params: %s", params)

    # Global importance over the last three years of training fights
    recent = ok & (year_of(rows) >= int(year_of(rows)[ok].max()) - 2)
    sv = fold(ensemble_shap(ens, design_matrix(rows[recent])))
    imp = sv.abs().mean().sort_values(ascending=False)
    imp.rename("mean_abs_shap").rename_axis("feature").to_csv(out / "shap_global.csv")
    plots.importance_bars(
        imp,
        "Global feature importance (LightGBM, fights 2024-2026)",
        out / "figures" / "shap_global.png",
    )

    # Upcoming card
    card = upcoming[upcoming["fight_id"].str.startswith(f"{args.event}#")].reset_index(drop=True)
    if card.empty:
        log.warning("No predictable bouts for %s", args.event)
        return
    xc = design_matrix(card)[cols]
    sym = symmetrize(card, ens.predict_proba(xc))
    sym["p_calibrated"] = cal.transform(sym["p"].to_numpy())
    raw = fold(ensemble_shap(ens, xc))
    raw.index = pd.MultiIndex.from_frame(card[["fight_id", "orientation"]])
    shap_sym = (raw.xs("AB", level="orientation") - raw.xs("BA", level="orientation")) / 2
    bouts, records = [], []
    order = sorted(sym.index, key=lambda f: int(f.split("#")[1]))
    for fid in order:
        s = sym.loc[fid]
        drivers = shap_sym.loc[fid].sort_values(key=np.abs, ascending=False)
        top = [(k, float(v)) for k, v in drivers.head(8).items()]
        bouts.append(
            {
                "a": s["fighter_name"],
                "b": s["opponent_name"],
                "p_a": float(s["p_calibrated"]),
                "drivers": top,
            }
        )
        records.append(
            {
                "bout": fid,
                "fighter_a": s["fighter_name"],
                "fighter_b": s["opponent_name"],
                "weight_class": s["weight_class"],
                "p_a_win": round(float(s["p_calibrated"]), 4),
                "p_a_uncalibrated": round(float(s["p"]), 4),
                "elo_p_a": round(float(s["elo_prob"]), 4),
                "top_drivers": "; ".join(f"{k} {v:+.3f}" for k, v in top[:5]),
            }
        )
    pd.DataFrame(records).to_csv(
        out / f"{args.event.replace(' ', '_').lower()}_predictions.csv", index=False
    )
    slug = args.event.replace(" ", "_").lower()
    plots.driver_grid(bouts, out / "figures" / f"{slug}_drivers.png", args.event)
    log.info("Wrote global importance and %d bout explanations for %s", len(bouts), args.event)


if __name__ == "__main__":
    main()
