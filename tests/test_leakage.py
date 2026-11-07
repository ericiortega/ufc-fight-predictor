"""Leakage tests: no feature for a fight may use data from that fight's date or later."""

from __future__ import annotations

import random
from datetime import date

import numpy as np
import pandas as pd
import pytest

from ufc_model.config import get_config
from ufc_model.features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FIGHTER_KEYS,
    NUMERIC_FEATURES,
    SYMMETRIC_COLUMNS,
    build_feature_table,
)

CFG = get_config()
COMPARE = FEATURE_COLUMNS + ["elo_prob", "meta_n_fights", "meta_opp_n_fights"]


def build(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_feature_table(t["fights"], t["rounds"], t["fighters"], CFG.features, CFG.elo).rows


def tamper_future(
    t: dict[str, pd.DataFrame], cutoff: date, seed: int = 0
) -> dict[str, pd.DataFrame]:
    """Scramble everything dated on/after ``cutoff``: results, methods, durations,
    round stats, extra and deleted fights, and profiles of fighters who debut after it."""
    rng = random.Random(seed)
    fights, rounds, fighters = t["fights"].copy(), t["rounds"].copy(), t["fighters"].copy()
    dates = pd.to_datetime(fights["event_date"]).dt.date
    future = dates >= cutoff
    n = int(future.sum())
    fights.loc[future, "winner_side"] = [rng.choice("AB") for _ in range(n)]
    fights.loc[future, "result"] = [rng.choice(["win", "win", "draw", "nc"]) for _ in range(n)]
    fights.loc[future, "method"] = [
        rng.choice(["KO/TKO", "Submission", "Decision - Split"]) for _ in range(n)
    ]
    fights["fight_duration_s"] = fights["fight_duration_s"].astype("float64")
    fights.loc[future, "fight_duration_s"] = [float(rng.randint(5, 1500)) for _ in range(n)]
    future_ids = set(fights.loc[future, "fight_id"])
    in_future = rounds["fight_id"].isin(future_ids)
    for col in ["sig_str_landed", "sig_str_attempted", "td_landed", "ctrl_s", "kd"]:
        rounds[col] = rounds[col].astype("float64")
        rounds.loc[in_future, col] = rounds.loc[in_future, col].map(lambda v: rng.randint(0, 99))
    # delete some future fights, add a fabricated one on the cutoff date itself
    fights = fights.drop(fights[future].sample(frac=0.3, random_state=seed).index)
    ids = fights["fighter_a_id"].dropna().unique()
    fake = fights.iloc[[0]].copy()
    is_ts = pd.api.types.is_datetime64_any_dtype(fights["event_date"])
    fake["fight_id"], fake["event_date"] = "FAKE", pd.Timestamp(cutoff) if is_ts else cutoff
    fake["fighter_a_id"], fake["fighter_b_id"] = ids[0], ids[1]
    fights = pd.concat([fights, fake], ignore_index=True)
    # profiles of fighters whose first fight is after the cutoff
    all_sides = pd.concat(
        [
            t["fights"][["event_date", "fighter_a_id"]].rename(columns={"fighter_a_id": "fid"}),
            t["fights"][["event_date", "fighter_b_id"]].rename(columns={"fighter_b_id": "fid"}),
        ]
    )
    first = all_sides.dropna().groupby("fid")["event_date"].min()
    # Strictly after: a debutant's own (static) profile legitimately feeds their own debut.
    later = set(first[pd.to_datetime(first).dt.date > cutoff].index)
    m = fighters["fighter_id"].isin(later)
    fighters.loc[m, "height_in"], fighters.loc[m, "reach_in"] = 90, 100
    fighters.loc[m, "stance"] = "Southpaw"
    fighters.loc[m, "dob"] = pd.Timestamp("1950-01-01")
    return {"fights": fights, "rounds": rounds, "fighters": fighters}


def assert_past_unchanged(before: pd.DataFrame, after: pd.DataFrame, cutoff: date) -> int:
    """Rows for fights dated <= cutoff must be identical; returns how many were compared."""
    key = ["fight_id", "orientation"]
    b = before[pd.to_datetime(before["event_date"]).dt.date <= cutoff].set_index(key).sort_index()
    a = after[pd.to_datetime(after["event_date"]).dt.date <= cutoff].set_index(key).sort_index()
    # Strictly before the cutoff nothing was added or deleted; on it, tampering may delete fights.
    strict_b = b[pd.to_datetime(b["event_date"]).dt.date < cutoff].index
    assert strict_b.isin(a.index).all()
    common = b.index.intersection(a.index)
    pd.testing.assert_frame_equal(b.loc[common, COMPARE], a.loc[common, COMPARE], check_exact=True)
    return len(common)


# ---------------------------------------------------------------- future tampering


def test_future_tampering_synthetic(league) -> None:
    dates = sorted(league["fights"]["event_date"].unique())
    for cutoff in (dates[5], dates[len(dates) // 2], dates[-3]):
        before = build(league)
        after = build(tamper_future(league, cutoff))
        assert assert_past_unchanged(before, after, cutoff) > 0
        # Control: the tampering is real, so later features *do* change.
        key = ["fight_id", "orientation"]
        late_b = before[pd.to_datetime(before["event_date"]).dt.date > cutoff].set_index(key)
        late_a = after.set_index(key).reindex(late_b.index).dropna(subset=["event_date"])
        if not late_a.empty:
            assert not late_b.loc[late_a.index, COMPARE].equals(late_a[COMPARE])


@pytest.mark.data
def test_future_tampering_real_data(real_tables) -> None:
    before = build(real_tables)
    cutoff = date(2016, 7, 9)  # UFC 200 weekend; also covers fights *on* the cutoff date
    n = assert_past_unchanged(before, build(tamper_future(real_tables, cutoff)), cutoff)
    assert n > 6000


def test_same_date_fights_do_not_see_each_other(league) -> None:
    """Event 3 books a fighter twice on one date; the second bout can't know the first."""
    rows = build(league)
    f = league["fights"]
    day = f[f["event_id"] == "e03"]
    a0 = day.iloc[0]["fighter_a_id"]
    mine = rows[(rows["fighter_id"] == a0) & rows["fight_id"].isin(day["fight_id"])]
    assert len(mine) == 2
    assert mine["meta_n_fights"].nunique() == 1


# ---------------------------------------------------------------- mirror / corner bias


def _check_mirror(rows: pd.DataFrame) -> None:
    ab = rows[rows["orientation"] == "AB"].set_index("fight_id").sort_index()
    ba = rows[rows["orientation"] == "BA"].set_index("fight_id").sort_index()
    assert ab.index.equals(ba.index), "every fight must have both orientations"
    diffs = [c for c in NUMERIC_FEATURES if c.startswith("diff_")]
    np.testing.assert_array_equal(ab[diffs].to_numpy(), -ba[diffs].to_numpy())
    pd.testing.assert_frame_equal(ab[SYMMETRIC_COLUMNS], ba[SYMMETRIC_COLUMNS])
    labeled = ab["label"].notna()
    assert ((ab.loc[labeled, "label"] + ba.loc[labeled, "label"]) == 1).all()
    assert ba.loc[~labeled, "label"].isna().all()
    np.testing.assert_allclose(ab["elo_prob"] + ba["elo_prob"], 1.0)
    assert (ab["event_date"] == ba["event_date"]).all(), (
        "mirror pair must share a date (same split)"
    )
    assert (ab["in_training"] == ba["in_training"]).all()
    assert (ab["fighter_id"].fillna("?") == ba["opponent_id"].fillna("?")).all()


def test_mirror_synthetic(league) -> None:
    _check_mirror(build(league))


@pytest.mark.data
def test_mirror_real_data(real_tables) -> None:
    rows = real_tables.get("fight_features")
    if rows is not None:
        assert not rows[NUMERIC_FEATURES].isna().any().any()
        assert not rows[CATEGORICAL_FEATURES].isna().any().any()
    _check_mirror(
        real_tables.get("fight_features") if "fight_features" in real_tables else build(real_tables)
    )


# ---------------------------------------------------------------- feature hygiene


def test_no_nan_and_no_missingness_features(league) -> None:
    rows = build(league)
    assert not rows[NUMERIC_FEATURES].isna().any().any()
    assert not rows[CATEGORICAL_FEATURES].isna().any().any()
    for col in FEATURE_COLUMNS:
        assert not any(
            w in col for w in ("missing", "imputed", "meta_", "label", "result", "elo_prob")
        )
    assert not any("missing" in k or "imputed" in k for k in FIGHTER_KEYS)


def test_profile_imputation_is_point_in_time(league) -> None:
    """Imputed values for a fight can only depend on fighters seen before it."""
    t = league
    first_date = sorted(t["fights"]["event_date"].unique())[0]
    rows = build(t)
    early = rows[pd.to_datetime(rows["event_date"]).dt.date == first_date]
    # On the first date nobody has been observed yet: defaults are used.
    for _, r in early.iterrows():
        if r["meta_missing_height"] and r["meta_opp_missing_height"]:
            assert r["diff_height_in"] == 0
    # Imputed heights / reaches are whole inches, like real ones.
    for col in ("diff_height_in", "diff_reach_in"):
        assert (rows[col] == rows[col].round()).all()


def test_training_rows_filter(league) -> None:
    rows = build(league)
    tr = rows[rows["in_training"]]
    assert (pd.to_datetime(tr["event_date"]).dt.year >= CFG.features.train_start_year).all()
    assert tr["label"].notna().all() and set(tr["result"]) == {"win"}
    assert tr["fighter_id"].notna().all() and tr["opponent_id"].notna().all()
    excluded = rows[~rows["in_training"]]
    assert {"draw", "nc"} <= set(excluded["result"].dropna())


def test_unresolved_fight_updates_resolved_opponent_only(league) -> None:
    f = league["fights"]
    bad = f[f["fighter_b_id"].isna()].iloc[0]
    rows = build(league)
    assert not rows[rows["fight_id"] == bad["fight_id"]]["in_training"].any()
    a = bad["fighter_a_id"]
    later = rows[
        (rows["fighter_id"] == a)
        & (pd.to_datetime(rows["event_date"]) > pd.Timestamp(bad["event_date"]))
    ]
    at = rows[(rows["fighter_id"] == a) & (rows["fight_id"] == bad["fight_id"])]
    if not later.empty:
        assert later.iloc[0]["meta_n_fights"] == at.iloc[0]["meta_n_fights"] + 1


def test_draws_and_ncs_count_as_fights_but_not_targets(league) -> None:
    rows = build(league)
    dn = rows[rows["result"].isin(["draw", "nc"])]
    assert dn["label"].isna().all() and not dn["in_training"].any()


# ---------------------------------------------------------------- profile-missingness (real data)


@pytest.mark.data
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Expected: the RAW source data leaks career length through profile missingness "
        "(missing reach -> short career). This is why the model never sees NaN or missingness "
        "flags. If this ever passes, the upstream data changed; re-examine the leak."
    ),
)
def test_raw_profile_missingness_adds_no_signal(real_tables) -> None:
    from ufc_model.features.diagnostics import raw_missingness_test

    rows = real_tables.get("fight_features")
    if rows is None:
        pytest.skip("fight_features not built")
    for r in raw_missingness_test(rows, tuple(CFG.leakage.test_years), CFG.project.seed):
        assert r.logloss_gain <= CFG.leakage.max_logloss_gain
        assert r.accuracy_gain <= CFG.leakage.max_accuracy_gain


@pytest.mark.data
def test_fed_profile_features_do_not_carry_missingness(real_tables) -> None:
    """The features we feed must not let a model recover the missingness leak."""
    from ufc_model.features.diagnostics import fed_feature_leak_test

    rows = real_tables.get("fight_features")
    if rows is None:
        pytest.skip("fight_features not built")
    r = fed_feature_leak_test(rows, tuple(CFG.leakage.test_years), CFG.project.seed)
    assert r.identifiability_gain <= CFG.leakage.max_identifiability_auc_gain
    assert r.captured_share <= CFG.leakage.max_captured_share
