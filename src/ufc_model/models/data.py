"""Design matrices, eligibility, time-based splits and symmetric predictions.

Splits are cut on event date, so both orientations of a fight always land on
the same side. Predictions are symmetrized per fight: p(A) = (p_AB + 1 - p_BA) / 2,
so p(A) + p(B) = 1 and corner order cannot matter.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd

from ufc_model.features.build import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from ufc_model.features.profile import STANCES
from ufc_model.weight_classes import WEIGHT_CLASS_LBS

# Fixed category levels (known a priori, not learned from data).
CATEGORY_LEVELS = {
    "weight_class": [*WEIGHT_CLASS_LBS, "Unknown"],
    "stance_matchup": ["-".join(sorted(p)) for p in combinations_with_replacement(STANCES, 2)],
}


def year_of(rows: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(rows["event_date"]).dt.year


def eligible(rows: pd.DataFrame, start_year: int) -> pd.Series:
    """Rows usable as training/evaluation targets: decided, both fighters resolved."""
    return (
        rows["label"].notna()
        & rows["fighter_id"].notna()
        & rows["opponent_id"].notna()
        & (year_of(rows) >= start_year)
    )


def design_matrix(rows: pd.DataFrame, drop: tuple[str, ...] = ()) -> pd.DataFrame:
    """Numeric features plus one-hot categoricals with fixed levels."""
    x = rows[[c for c in NUMERIC_FEATURES if c not in drop]].astype(float)
    for col in CATEGORICAL_FEATURES:
        if col in drop:
            continue
        unknown = ~rows[col].isin(CATEGORY_LEVELS[col])
        if unknown.any():
            raise ValueError(f"Unknown {col} level(s): {set(rows.loc[unknown, col])}")
        cat = pd.Categorical(rows[col], categories=CATEGORY_LEVELS[col])
        dummies = pd.get_dummies(cat, prefix=col, dtype=float)
        dummies.index = rows.index
        x = pd.concat([x, dummies], axis=1)
    return x


def symmetrize(rows: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Per-fight symmetric probability for the AB fighter, with AB-row metadata."""
    df = rows[["fight_id", "orientation"]].assign(p=p)
    wide = df.pivot(index="fight_id", columns="orientation", values="p")
    if {"AB", "BA"} - set(wide.columns) or wide.isna().any().any():
        raise ValueError("Every fight needs both orientations to symmetrize")
    sym = (wide["AB"] + 1 - wide["BA"]) / 2
    ab = rows[rows["orientation"] == "AB"].set_index("fight_id")
    out = ab.loc[sym.index].copy()
    out["p"] = sym
    return out


def mirror(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Both orientations of per-fight probabilities (for fitting symmetric calibrators)."""
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])


@dataclass(frozen=True)
class Window:
    """One walk-forward step. All masks index the full feature table."""

    test_year: int
    train: pd.Series  # eligible, year < test_year
    inner_fit: pd.Series  # train AND year < test_year - inner_val_years
    inner_val: pd.Series  # train AND the last inner_val_years years
    test: pd.Series  # eligible, year == test_year


def walk_forward(
    rows: pd.DataFrame, start_year: int, test_years: tuple[int, int], inner_val_years: int
) -> Iterator[Window]:
    """Expanding-window splits: train strictly on the past, test on the next year."""
    year = year_of(rows)
    ok = eligible(rows, start_year)
    for y in range(test_years[0], test_years[1] + 1):
        test = ok & (year == y)
        if not test.any():
            continue
        train = ok & (year < y)
        cut = y - inner_val_years
        yield Window(y, train, train & (year < cut), train & (year >= cut), test)
