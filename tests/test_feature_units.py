"""Unit tests for Elo, history rates and point-in-time profile imputation."""

from datetime import date

import pytest

from ufc_model.features.elo import EloParams, EloRatings, expected_score
from ufc_model.features.history import (
    FighterHistory,
    FightRecord,
    GlobalPriors,
    ShrinkConfig,
    Totals,
    method_category,
)
from ufc_model.features.profile import Profile, ProfileImputer, stance_category

K = ShrinkConfig(15, 40, 6)


def test_elo_expected_and_zero_sum() -> None:
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) == pytest.approx(1 - expected_score(1400, 1600))
    elo = EloRatings(EloParams(1500, 80, 1.0))
    ra, rb = elo.update("a", "b", 1.0, finish=False)
    assert (ra, rb) == (1540, 1460)
    assert ra + rb == 3000


def test_elo_draw_and_finish_multiplier() -> None:
    elo = EloRatings(EloParams(1500, 32, 1.5))
    assert elo.update("a", "b", 0.5, finish=False) == (1500, 1500)
    ra, _ = elo.update("a", "b", 1.0, finish=True)
    assert ra == pytest.approx(1500 + 32 * 1.5 * 0.5)


def test_elo_unresolved_side_is_neutral_and_not_stored() -> None:
    elo = EloRatings(EloParams(1500, 80, 1.0))
    elo.update("a", None, 1.0, finish=False)
    assert elo.get("a") == 1540 and None not in elo.ratings and elo.get(None) == 1500


@pytest.mark.parametrize(
    ("raw", "cat"),
    [
        ("KO/TKO", "ko"),
        ("TKO - Doctor's Stoppage", "ko"),
        ("Submission", "sub"),
        ("Decision - Split", "dec"),
        ("DQ", "other"),
        (None, "other"),
    ],
)
def test_method_category(raw, cat) -> None:
    assert method_category(raw) == cat


def test_history_streaks_and_shrinkage() -> None:
    h = FighterHistory()
    t = Totals(minutes=15, sig_l=60, sig_a=120, opp_sig_l=30, opp_sig_a=100)
    for i, (o, m) in enumerate(
        [("W", "ko"), ("L", "dec"), ("W", "sub"), ("W", "dec"), ("NC", "other")]
    ):
        h.add(FightRecord(date(2020, 1, 1 + i), o, m, t))
    priors = GlobalPriors()
    priors.observe(Totals(minutes=1000, sig_l=3500, sig_a=8000, opp_sig_l=3500, opp_sig_a=8000))
    f = h.features(date(2020, 3, 1), priors, K, 3)
    assert f["n_fights"] == 5 and f["n_wins"] == 3 and f["n_losses"] == 1
    assert f["streak"] == 2  # NC skipped, then W, W, then the L stops it
    assert f["layoff_days"] == (date(2020, 3, 1) - date(2020, 1, 5)).days
    # 300 landed over 75 min = 4.0/min raw, pulled slightly toward the 3.5 prior
    assert 3.5 < f["slpm"] < 4.0


def test_debut_features_use_priors() -> None:
    priors = GlobalPriors()
    priors.observe(Totals(minutes=100, sig_l=350, sig_a=800))
    f = FighterHistory().features(date(2020, 1, 1), priors, K, 3)
    assert f["is_debut"] == 1 and f["n_fights"] == 0 and f["layoff_days"] == 0
    assert f["slpm"] == pytest.approx(3.5) and f["sig_acc"] == pytest.approx(350 / 800)


def test_imputer_uses_only_observed_fighters_and_rounds() -> None:
    imp = ProfileImputer(70, 71, 28, "orthodox")
    unknown = Profile(None, None, None, None)
    d = date(2020, 1, 1)
    first = imp.impute(unknown, "Lightweight", d)
    assert (first.height_in, first.reach_in, first.stance, first.age_years) == (
        70,
        71,
        "orthodox",
        28,
    )
    assert all(first.imputed.values())
    imp.observe("x", Profile(68, 71, "southpaw", date(1990, 1, 1)), "Lightweight", d)
    imp.observe("y", Profile(70, 73, "southpaw", date(1994, 1, 1)), "Lightweight", d)
    later = imp.impute(unknown, "Lightweight", date(2020, 6, 1))
    assert later.height_in == 69 and later.reach_in == 72 and later.stance == "southpaw"
    assert later.age_years == pytest.approx(28, abs=0.1)  # median age at fight time (30, 26)
    # Height known, reach missing: height + mean ape offset (+3), whole inches
    r = imp.impute(Profile(71, None, "orthodox", None), "Lightweight", date(2020, 6, 1))
    assert r.reach_in == 74 and r.imputed == {
        "height": False,
        "reach": True,
        "stance": False,
        "age": True,
    }


def test_imputer_ignores_repeat_observations_of_same_fighter() -> None:
    imp = ProfileImputer(70, 71, 28, "orthodox")
    for _ in range(5):
        imp.observe("x", Profile(80, 82, "switch", None), "Heavyweight", date(2020, 1, 1))
    imp.observe("y", Profile(74, 76, "orthodox", None), "Heavyweight", date(2020, 1, 1))
    imp.observe("z", Profile(76, 78, "orthodox", None), "Heavyweight", date(2020, 1, 1))
    assert (
        imp.impute(Profile(None, None, None, None), "Heavyweight", date(2021, 1, 1)).height_in == 76
    )


def test_stance_category() -> None:
    assert stance_category("Orthodox") == "orthodox"
    assert stance_category("Open Stance") == "other"
    assert stance_category(None) is None and stance_category(float("nan")) is None
