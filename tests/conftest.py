"""Shared fixtures: a small synthetic league, and the real DuckDB when present."""

from __future__ import annotations

import random
from datetime import date, timedelta

import pandas as pd
import pytest

from ufc_model.config import get_config

WEIGHT_CLASSES = ["Lightweight", "Welterweight", "Women's Strawweight"]


def make_league(seed: int = 7, n_fighters: int = 48, n_events: int = 36) -> dict[str, pd.DataFrame]:
    """Synthetic fights/rounds/fighters shaped like the real tables.

    Includes draws, no-contests, an unresolved fighter, a fighter booked twice on
    one date (early tournament style), pre-2005 fights, and missing profile fields.
    """
    rng = random.Random(seed)
    fighters = []
    for i in range(n_fighters):
        h = rng.randint(64, 76)
        fighters.append(
            {
                "fighter_id": f"f{i:02d}",
                "name": f"Fighter {i:02d}",
                "height_in": None if i % 11 == 0 else h,
                "reach_in": None if i % 5 == 0 else h + rng.randint(-1, 4),
                "stance": None
                if i % 9 == 0
                else rng.choice(["Orthodox", "Orthodox", "Southpaw", "Switch"]),
                "dob": None
                if i % 13 == 0
                else date(1975 + rng.randint(0, 25), rng.randint(1, 12), 1),
                "weight_lbs": 155.0,
            }
        )
    fights, rounds = [], []
    day = date(2003, 1, 4)
    for e in range(n_events):
        day += timedelta(days=rng.randint(20, 120))
        pool = rng.sample(range(n_fighters), 10)
        pairs = [(pool[i], pool[i + 1]) for i in range(0, 10, 2)]
        if e == 3:  # same fighter twice on one date
            pairs.append((pool[0], pool[2]))
        for j, (a, b) in enumerate(pairs):
            fid = f"e{e:02d}_{j}"
            r = rng.random()
            result, winner = (
                ("draw", None)
                if r < 0.05
                else ("nc", None)
                if r < 0.08
                else ("win", rng.choice("AB"))
            )
            method = rng.choice(
                ["KO/TKO", "Submission", "Decision - Unanimous", "Decision - Split"]
            )
            end_round = rng.randint(1, 3)
            end_time = 300 if method.startswith("Decision") else rng.randint(10, 300)
            fights.append(
                {
                    "fight_id": fid,
                    "event_id": f"e{e:02d}",
                    "event_date": day,
                    "fighter_a_id": f"f{a:02d}",
                    "fighter_b_id": None if (e == 20 and j == 0) else f"f{b:02d}",  # unresolved
                    "fighter_a_name": f"Fighter {a:02d}",
                    "fighter_b_name": f"Fighter {b:02d}",
                    "result": result,
                    "winner_side": winner,
                    "weight_class": rng.choice(WEIGHT_CLASSES),
                    "is_title": j == 0 and e % 6 == 0,
                    "is_women": False,
                    "method": method,
                    "scheduled_rounds": 5 if j == 0 else 3,
                    "fight_duration_s": (end_round - 1) * 300 + end_time,
                }
            )
            if e % 10 == 1:
                continue  # some fights have no round stats
            for side in "AB":
                for rd in range(1, end_round + 1):
                    att = rng.randint(10, 60)
                    rounds.append(
                        {
                            "fight_id": fid,
                            "side": side,
                            "round": rd,
                            "sig_str_landed": rng.randint(0, att),
                            "sig_str_attempted": att,
                            "td_landed": rng.randint(0, 2),
                            "td_attempted": rng.randint(2, 4),
                            "sub_att": rng.randint(0, 1),
                            "ctrl_s": rng.randint(0, 120),
                            "kd": rng.randint(0, 1),
                        }
                    )
    return {
        "fights": pd.DataFrame(fights),
        "rounds": pd.DataFrame(rounds),
        "fighters": pd.DataFrame(fighters),
    }


@pytest.fixture
def league() -> dict[str, pd.DataFrame]:
    return make_league()


@pytest.fixture(scope="session")
def real_tables() -> dict[str, pd.DataFrame]:
    """Real tables from DuckDB; tests using this skip if the database is absent."""
    path = get_config().paths.db_path
    if not path.exists():
        pytest.skip("No local DuckDB; run `python -m ufc_model.ingest` first")
    import duckdb

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {
            n: con.sql(f"select * from {t}").df()
            for n, t in (("fights", "fights"), ("rounds", "fight_rounds"), ("fighters", "fighters"))
        }
        has_features = "fight_features" in {r[0] for r in con.sql("show tables").fetchall()}
        if has_features:
            tables["fight_features"] = con.sql("select * from fight_features").df()
    finally:
        con.close()
    return tables
