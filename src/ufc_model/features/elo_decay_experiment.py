"""Report-only experiment: Elo with decay toward the mean after long layoffs.

``python -m ufc_model.features.elo_decay_experiment``

Before a fighter's next fight, if their layoff exceeds ``grace`` years, their
rating is pulled toward the initial rating:
    r <- mean + (r - mean) * exp(-lam * (layoff_years - grace))
``lam`` and ``grace`` are tuned on 2005-2012 only; 2013-2026 is reported once
with and without decay. Production Elo is unchanged.
"""

from __future__ import annotations

import itertools
import math
from datetime import date

import pandas as pd

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.features.elo import expected_score

LAMBDAS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5]
GRACES = [0.5, 1.0, 1.5]


def score(
    fights: pd.DataFrame,
    k: float,
    lam: float,
    grace: float,
    start: date,
    end: date,
    mean: float = 1500.0,
) -> tuple[float, float, int]:
    ratings: dict[str, float] = {}
    last: dict[str, date] = {}
    loss = hits = 0.0
    n = 0
    for on, day in fights.sort_values("event_date").groupby("event_date", sort=True):
        on = pd.Timestamp(on).date()
        recs = day.to_dict("records")

        def rating(fid):
            if not isinstance(fid, str):
                return mean
            r = ratings.get(fid, mean)
            if fid in last and lam > 0:
                gap = (on - last[fid]).days / 365.25  # noqa: B023 - used within this iteration
                if gap > grace:
                    r = mean + (r - mean) * math.exp(-lam * (gap - grace))
            return r

        pre = {}
        for f in recs:
            for fid in (f["fighter_a_id"], f["fighter_b_id"]):
                if isinstance(fid, str):
                    pre[fid] = rating(fid)
        if start <= on <= end:
            for f in recs:
                a, b = f["fighter_a_id"], f["fighter_b_id"]
                if f["result"] != "win" or not isinstance(a, str) or not isinstance(b, str):
                    continue
                p = min(max(expected_score(pre[a], pre[b]), 1e-6), 1 - 1e-6)
                y = f["winner_side"] == "A"
                loss -= math.log(p if y else 1 - p)
                hits += 0.5 if p == 0.5 else float((p > 0.5) == y)
                n += 1
        for f in recs:
            a, b = f["fighter_a_id"], f["fighter_b_id"]
            ids = [x if isinstance(x, str) else None for x in (a, b)]
            for fid in ids:
                if fid is not None:
                    ratings[fid] = pre[fid]  # decay is applied when they return
                    last[fid] = on
            if f["result"] not in {"win", "draw"} or ids == [None, None]:
                continue
            ra, rb = (pre[i] if i else mean for i in ids)
            s = 0.5 if f["result"] == "draw" else float(f["winner_side"] == "A")
            d = k * (s - expected_score(ra, rb))
            if ids[0]:
                ratings[ids[0]] = ra + d
            if ids[1]:
                ratings[ids[1]] = rb - d
    return loss / n, hits / n, n


def main() -> None:
    cfg = get_config()
    con = db.connect(cfg.paths.db_path)
    fights = con.sql("select * from fights").df()
    con.close()
    k = cfg.elo.k
    tune = [
        (lam, g, *score(fights, k, lam, g, cfg.elo.tuning_start, cfg.elo.tuning_end))
        for lam, g in itertools.product(LAMBDAS, GRACES)
    ]
    tune.sort(key=lambda t: t[2])
    best_lam, best_grace = tune[0][0], tune[0][1]
    test_start, test_end = date(2013, 1, 1), date(2026, 12, 31)
    rows = []
    for label, lam, g in (("no decay", 0.0, 1.0), ("decay", best_lam, best_grace)):
        tl, ta, tn = score(fights, k, lam, g, cfg.elo.tuning_start, cfg.elo.tuning_end)
        ll, acc, n = score(fights, k, lam, g, test_start, test_end)
        rows.append(
            {
                "variant": label,
                "lambda": lam,
                "grace_years": g,
                "tune_logloss": tl,
                "tune_accuracy": ta,
                "test_logloss": ll,
                "test_accuracy": acc,
                "test_n": n,
            }
        )
    out = cfg.models.reports_dir.parent / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        tune, columns=["lambda", "grace_years", "tune_logloss", "tune_accuracy", "n"]
    ).to_csv(out / "elo_decay_grid.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(out / "elo_decay.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
