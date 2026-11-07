"""Compile ``reports/phase3/summary.md`` from the evaluation CSVs.

``python -m ufc_model.models.report``
"""

from __future__ import annotations

import pandas as pd

from ufc_model.config import get_config

LABELS = {
    "elo": "Elo",
    "higher_win_rate": "Higher win rate",
    "more_ufc_fights": "More UFC fights",
    "younger": "Younger fighter",
    "lr": "Logistic regression",
    "lgbm": "LightGBM (5-seed)",
    "lgbm_no_profile": "LightGBM without height/reach/age",
    "lr_start2001": "Logistic regression, train from 2001",
    "lgbm_start2001": "LightGBM, train from 2001",
}


def _fmt(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in ("accuracy",):
        if c in df:
            df[c] = df[c].map(lambda v: f"{v:.1%}")
    for c in ("log_loss", "brier"):
        if c in df:
            df[c] = df[c].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
    if "model" in df:
        df["model"] = df["model"].map(lambda m: LABELS.get(m, m))
    return df.to_markdown(index=False)


def main() -> None:
    cfg = get_config()
    d = cfg.models.reports_dir
    pooled = pd.read_csv(d / "metrics_pooled.csv")
    yearly = pd.read_csv(d / "metrics_by_year.csv")
    subsets = pd.read_csv(d / "metrics_subsets.csv")
    best_cal = "none"  # uncalibrated scored best pooled; see the calibration rows
    main_models = ["elo", "higher_win_rate", "more_ufc_fights", "younger", "lr", "lgbm"]

    lines = [
        "# Phase 3: walk-forward evaluation (2013-2026)",
        "",
        "Predictions and research findings only; not betting advice.",
        "",
        "Expanding window: each test year is predicted by models trained only on earlier "
        "years. Optuna tuning and calibration use the last two years of each training "
        "window. Probabilities are symmetric per fight.",
        "",
    ]

    lines += ["## Pooled results", ""]
    p = pooled[pooled["model"].isin(main_models)]
    lines.append(
        _fmt(
            p.sort_values(["model", "calibration"])[
                ["model", "calibration", "n", "accuracy", "log_loss", "brier"]
            ]
        )
    )
    lines += [
        "",
        "Pick-only baselines have no probabilities, so only accuracy is reported. "
        "Platt scaling without an intercept (required for symmetry) is mathematically "
        "the same as temperature scaling.",
        "",
    ]

    lines += ["## By test year (uncalibrated)", ""]
    y = yearly[
        (yearly["model"].isin(["elo", "lr", "lgbm"]))
        & ((yearly["calibration"] == best_cal) | (yearly["model"] == "elo"))
    ]
    wide = y.pivot_table(index="test_year", columns="model", values=["accuracy", "log_loss"])
    wide.columns = [f"{LABELS[m]} {metric.replace('_', ' ')}" for metric, m in wide.columns]
    lines.append(wide.reset_index().round(4).to_markdown(index=False))
    lines.append("")

    lines += ["## Subsets", ""]
    s = subsets[
        subsets["model"].isin(["elo", "lr", "lgbm"])
        & ((subsets["calibration"] == best_cal) | (subsets["model"] == "elo"))
    ]
    lines.append(_fmt(s[["subset", "model", "n", "accuracy", "log_loss", "brier"]]))
    lines.append("")

    lines += ["## Profile-feature check (LightGBM, same tuned parameters per year)", ""]
    pc = pooled[
        pooled["model"].isin(["lgbm", "lgbm_no_profile"]) & (pooled["calibration"] == best_cal)
    ]
    lines.append(_fmt(pc[["model", "n", "accuracy", "log_loss", "brier"]]))
    lines.append("")

    lines += ["## Training-start experiment (2001 vs 2005)", ""]
    se = pooled[
        pooled["model"].isin(["lr", "lgbm", "lr_start2001", "lgbm_start2001"])
        & (pooled["calibration"] == best_cal)
    ]
    lines.append(_fmt(se[["model", "n", "accuracy", "log_loss", "brier"]]))
    lines.append("")

    imp_path = d / "shap_global.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path, index_col=0)["mean_abs_shap"].head(15)
        lines += [
            "## Global SHAP importance (top 15)",
            "",
            imp.round(4)
            .reset_index()
            .rename(columns={"index": "feature"})
            .to_markdown(index=False),
            "",
        ]
    for f in sorted(d.glob("*_predictions.csv")):
        card = pd.read_csv(f)
        lines += [
            f"## {f.stem.replace('_predictions', '').replace('_', ' ').upper()} "
            "predicted win probabilities",
            "",
            card[
                ["fighter_a", "fighter_b", "weight_class", "p_a_win", "elo_p_a", "top_drivers"]
            ].to_markdown(index=False),
            "",
        ]
    lines += ["## Figures", ""] + [
        f"- `figures/{p.name}`" for p in sorted((d / "figures").glob("*.png"))
    ]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {d / 'summary.md'}")


if __name__ == "__main__":
    main()
