"""Splits, symmetric calibration, and walk-forward isolation of the test year."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufc_model.config import get_config
from ufc_model.features.build import build_feature_table
from ufc_model.models.calibration import CALIBRATORS, make_calibrator
from ufc_model.models.data import design_matrix, symmetrize, walk_forward, year_of
from ufc_model.models.evaluate import (
    LeakageAlarm,
    accuracy,
    check_leakage_alarm,
    run_walk_forward,
)


@pytest.fixture
def rows(league) -> pd.DataFrame:
    cfg = get_config()
    out = build_feature_table(
        league["fights"], league["rounds"], league["fighters"], cfg.features, cfg.elo
    ).rows
    return out.sort_values(["event_date", "fight_id", "orientation"]).reset_index(drop=True)


@pytest.fixture
def small_cfg():
    cfg = get_config().model_copy(deep=True)
    cfg.models.test_years = (2009, 2010)
    cfg.models.optuna_trials_lgbm = cfg.models.optuna_trials_lr = 2
    cfg.models.lgbm_seeds = [1, 2]
    return cfg


def test_walk_forward_windows(rows) -> None:
    year = year_of(rows)
    windows = list(walk_forward(rows, 2003, (2007, 2010), 2))
    present = sorted(set(year[rows["label"].notna() & year.between(2007, 2010)]))
    assert [w.test_year for w in windows] == present and len(present) >= 2
    for w in windows:
        assert (year[w.train] < w.test_year).all()
        assert (year[w.test] == w.test_year).all()
        assert (year[w.inner_val] >= w.test_year - 2).all() and (
            year[w.inner_fit] < w.test_year - 2
        ).all()
        assert (
            not (w.inner_fit & w.inner_val).any() and ((w.inner_fit | w.inner_val) == w.train).all()
        )
        for mask in (w.train, w.test, w.inner_fit, w.inner_val):
            per_fight = rows[mask].groupby("fight_id")["orientation"].nunique()
            assert (per_fight == 2).all(), "both orientations of a fight share a split"
        assert rows[w.test]["label"].notna().all()


def test_design_matrix_has_fixed_columns(rows) -> None:
    a = design_matrix(rows.iloc[:20])
    b = design_matrix(rows)
    assert list(a.columns) == list(b.columns)
    assert not b.isna().any().any()
    bad = rows.iloc[:2].copy()
    bad["weight_class"] = "Cruiserweight"
    with pytest.raises(ValueError, match="Unknown weight_class"):
        design_matrix(bad)


def test_symmetrize(rows) -> None:
    sub = rows[rows["label"].notna()].iloc[:40]
    p = np.where(sub["orientation"] == "AB", 0.7, 0.4)  # p_AB=0.7, p_BA=0.4
    s = symmetrize(sub, p)
    np.testing.assert_allclose(s["p"], (0.7 + 0.6) / 2)
    with pytest.raises(ValueError):
        symmetrize(
            sub[sub["orientation"] == "AB"], np.full((sub["orientation"] == "AB").sum(), 0.5)
        )


@pytest.mark.parametrize("name", list(CALIBRATORS))
def test_calibrators_preserve_symmetry(name) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 600)
    y = (rng.uniform(size=600) < 0.5 + 0.6 * (p - 0.5)).astype(int)
    cal = make_calibrator(name).fit(p, y)
    grid = np.linspace(0.02, 0.98, 49)
    np.testing.assert_allclose(cal.transform(grid) + cal.transform(1 - grid), 1.0, atol=1e-9)
    out = cal.transform(grid)
    assert ((out > 0) & (out < 1)).all()


def test_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(1)
    true = rng.uniform(0.2, 0.8, 20000)
    y = (rng.uniform(size=true.size) < true).astype(int)
    logit = np.log(true / (1 - true))
    overconfident = 1 / (1 + np.exp(-2 * logit))  # T should come back near 2
    t = make_calibrator("temperature").fit(overconfident, y).t
    assert 1.8 < t < 2.2


def test_accuracy_half_credit_for_coin_flips() -> None:
    assert accuracy(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert accuracy(np.array([1, 0, 1]), np.array([0.9, 0.2, 0.3])) == pytest.approx(2 / 3)


def test_leakage_alarm_trips() -> None:
    preds = pd.DataFrame(
        {
            "model": "m",
            "calibration": "none",
            "test_year": 2020,
            "label": [1, 0] * 50,
            "p": [0.9, 0.1] * 50,
        }
    )
    with pytest.raises(LeakageAlarm):
        check_leakage_alarm(preds, 0.75)
    check_leakage_alarm(preds.assign(p=0.5), 0.75)


def test_test_year_labels_never_reach_training(rows, small_cfg) -> None:
    """Scramble the labels of each test year: that year's predictions must not move."""
    before = run_walk_forward(rows, small_cfg, 2003).preds
    for year in range(small_cfg.models.test_years[0], small_cfg.models.test_years[1] + 1):
        scrambled = rows.copy()
        in_year = (year_of(rows) == year) & rows["label"].notna()
        ab = in_year & (rows["orientation"] == "AB")
        flip = rows.loc[ab, "fight_id"].sample(frac=0.5, random_state=0)
        hit = in_year & rows["fight_id"].isin(flip)
        scrambled.loc[hit, "label"] = 1 - scrambled.loc[hit, "label"]
        after = run_walk_forward(scrambled, small_cfg, 2003).preds
        key = ["fight_id", "model", "calibration"]
        b = before[before["test_year"] == year].set_index(key)["p"].sort_index()
        a = after[after["test_year"] == year].set_index(key)["p"].sort_index()
        pd.testing.assert_series_equal(b, a)
