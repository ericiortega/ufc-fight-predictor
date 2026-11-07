"""Grid-search Elo settings: ``python -m ufc_model.features.tune_elo``.

Ratings run over *all* fights from 1994, but the score (log loss of the Elo win
probability) is computed only on fights inside the tuning window, which ends
before any walk-forward evaluation year. Results are printed; copy the winner
into ``config/config.yaml``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.features.elo import EloParams, EloRatings, expected_score
from ufc_model.features.history import method_category
from ufc_model.logging_setup import setup_logging


@dataclass(frozen=True)
class EloScore:
    k: float
    finish_multiplier: float
    log_loss: float
    accuracy: float
    n: int


def score_elo(fights: pd.DataFrame, params: EloParams, start: date, end: date) -> EloScore:
    """Pre-fight Elo log loss / accuracy on decided, fully resolved fights in [start, end]."""
    elo = EloRatings(params)
    losses, hits, n = 0.0, 0.0, 0
    for on, day in fights.sort_values("event_date").groupby("event_date", sort=True):
        on = pd.Timestamp(on).date()
        recs = day.to_dict("records")
        if start <= on <= end:
            for f in recs:
                a, b = f["fighter_a_id"], f["fighter_b_id"]
                if f["result"] != "win" or not isinstance(a, str) or not isinstance(b, str):
                    continue
                p = expected_score(elo.get(a), elo.get(b))
                y = f["winner_side"] == "A"
                p = min(max(p, 1e-6), 1 - 1e-6)
                losses -= math.log(p if y else 1 - p)
                hits += 0.5 if p == 0.5 else float((p > 0.5) == y)  # coin flip: half credit
                n += 1
        for f in recs:
            if f["result"] not in {"win", "draw"}:
                continue
            ids = [
                x if isinstance(x, str) else None for x in (f["fighter_a_id"], f["fighter_b_id"])
            ]
            if ids == [None, None]:
                continue
            score = 0.5 if f["result"] == "draw" else float(f["winner_side"] == "A")
            elo.update(*ids, score, method_category(f["method"]) in {"ko", "sub"})
    return EloScore(params.k, params.finish_multiplier, losses / n, hits / n, n)


def main() -> None:
    setup_logging()
    cfg = get_config()
    con = db.connect(cfg.paths.db_path)
    fights = con.sql("select * from fights").df()
    con.close()
    e = cfg.elo
    results = [
        score_elo(fights, EloParams(e.initial, k, fm), e.tuning_start, e.tuning_end)
        for k, fm in itertools.product(e.grid_k, e.grid_finish_multiplier)
    ]
    results.sort(key=lambda r: r.log_loss)
    print(f"Tuning window {e.tuning_start} .. {e.tuning_end}, n={results[0].n} fights")
    print(f"{'K':>6} {'finish_x':>9} {'log_loss':>9} {'accuracy':>9}")
    for r in results:
        print(f"{r.k:6.0f} {r.finish_multiplier:9.2f} {r.log_loss:9.4f} {r.accuracy:9.3f}")
    best = results[0]
    print(f"\nBest: k={best.k:g}, finish_multiplier={best.finish_multiplier:g}")


if __name__ == "__main__":
    main()
