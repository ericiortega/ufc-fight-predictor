"""Static report figures (matplotlib).

Style follows a validated reference palette: fixed categorical slot order,
hairline solid gridlines, 2px lines, direct labels plus a legend, one y-axis per
chart, blue/red poles for signed contributions. Every figure has a CSV table
beside it in ``reports/`` (two slots sit below 3:1 contrast, so labels and
tables carry identity, not color alone).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # slots 1-4, fixed order
POS, NEG = "#2a78d6", "#e34948"  # diverging poles

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    }
)


def reliability(
    preds: pd.DataFrame,
    series: list[tuple[str, str, str]],
    title: str,
    path: Path,
    bins: int = 10,
    min_n: int = 30,
) -> pd.DataFrame:
    """Reliability curve on mirrored per-fight probabilities.

    ``series``: (label, model, calibration) triples, drawn in fixed slot order.
    """
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    ax.annotate("perfect calibration", (0.82, 0.86), color=MUTED, fontsize=8, rotation=42)
    table = []
    for (label, model, cal), color in zip(series, SERIES, strict=False):
        g = preds[(preds["model"] == model) & (preds["calibration"] == cal)]
        p = np.concatenate([g["p"], 1 - g["p"]])
        y = np.concatenate([g["label"], 1 - g["label"]])
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
        pts = [
            (p[idx == b].mean(), y[idx == b].mean(), int((idx == b).sum()))
            for b in range(bins)
            if (idx == b).sum() >= min_n
        ]
        xs, ys, ns = map(np.array, zip(*pts, strict=True))
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
        )
        table += [
            {"series": label, "mean_predicted": a, "observed": b, "n": n}
            for a, b, n in zip(xs, ys, ns, strict=True)
        ]
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted win probability",
        ylabel="Observed win rate",
        title=title,
    )
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    df = pd.DataFrame(table)
    df.to_csv(path.with_suffix(".csv"), index=False)
    return df


def per_year(
    table: pd.DataFrame,
    metric: str,
    series: list[tuple[str, str, str]],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ends = []
    for (label, model, cal), color in zip(series, SERIES, strict=False):
        g = table[(table["model"] == model) & (table["calibration"] == cal)].sort_values(
            "test_year"
        )
        ax.plot(
            g["test_year"],
            g[metric],
            "-o",
            color=color,
            label=label,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        ends.append([g["test_year"].iloc[-1], g[metric].iloc[-1], label, color])
    # Nudge end labels apart so close lines don't print on top of each other.
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort(key=lambda e: e[1])
    label_y = [e[1] for e in ends]
    for i in range(1, len(label_y)):
        label_y[i] = max(label_y[i], label_y[i - 1] + gap)
    for (x, y, label, color), ly in zip(ends, label_y, strict=True):
        ax.plot([x], [y], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.annotate(
            label,
            (x, ly),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
        )
    ax.set(title=title, xlabel="Test year", ylabel=ylabel)
    ax.set_xticks(sorted(table["test_year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best")
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def importance_bars(imp: pd.Series, title: str, path: Path, top: int = 20) -> None:
    imp = imp.sort_values(ascending=True).tail(top)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.values, color=SERIES[0], height=0.62)
    ax.grid(axis="y", visible=False)
    ax.set(title=title, xlabel="Mean |SHAP| (log-odds)")
    for i, v in enumerate(imp.values):
        ax.annotate(
            f"{v:.3f}",
            (v, i),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK_2,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def driver_grid(bouts: list[dict], path: Path, event: str, top: int = 6) -> None:
    """Small multiples: top signed SHAP drivers per bout (positive favors fighter A)."""
    n = len(bouts)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.3 * rows), squeeze=False)
    for ax, b in zip(axes.flat, bouts, strict=False):
        d = b["drivers"][:top][::-1]
        vals = [v for _, v in d]
        ax.barh([k for k, _ in d], vals, color=[POS if v > 0 else NEG for v in vals], height=0.6)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.grid(axis="y", visible=False)
        lim = max(0.05, max(abs(v) for v in vals) * 1.15)
        ax.set_xlim(-lim, lim)
        ax.set_title(
            f"{b['a']} {b['p_a']:.0%}  vs  {b['b']} {1 - b['p_a']:.0%}", fontsize=10, loc="left"
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("SHAP, log-odds (+ favors first-named)", fontsize=8, color=MUTED)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(
        f"{event}: top drivers per bout (blue favors the first-named fighter, red the second)",
        fontsize=11,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(preds: pd.DataFrame, fig_dir: Path) -> None:
    from ufc_model.models.evaluate import metric_table

    main = preds[~preds["model"].str.contains("_")]
    reliability(
        main,
        [
            ("LightGBM, uncalibrated", "lgbm", "none"),
            ("LightGBM + Platt / temperature", "lgbm", "platt"),
            ("LightGBM + isotonic", "lgbm", "isotonic"),
        ],
        "Reliability: LightGBM calibrators (2013-2026; Platt = temperature here)",
        fig_dir / "reliability_lgbm.png",
    )
    reliability(
        main,
        [
            ("Logistic, uncalibrated", "lr", "none"),
            ("Logistic + Platt / temperature", "lr", "platt"),
            ("Logistic + isotonic", "lr", "isotonic"),
        ],
        "Reliability: logistic calibrators (2013-2026; Platt = temperature here)",
        fig_dir / "reliability_lr.png",
    )
    reliability(
        main,
        [
            ("Elo", "elo", "none"),
            ("Logistic", "lr", "none"),
            ("LightGBM", "lgbm", "none"),
        ],
        "Reliability: Elo vs models, uncalibrated (2013-2026)",
        fig_dir / "reliability_models.png",
    )
    yearly = metric_table(main, ["model", "calibration", "test_year"])
    series = [("Elo", "elo", "none"), ("Logistic", "lr", "none"), ("LightGBM", "lgbm", "none")]
    per_year(
        yearly,
        "accuracy",
        series,
        "Accuracy by test year",
        "Accuracy",
        fig_dir / "accuracy_by_year.png",
    )
    per_year(
        yearly,
        "log_loss",
        series,
        "Log loss by test year (lower is better)",
        "Log loss",
        fig_dir / "logloss_by_year.png",
    )
