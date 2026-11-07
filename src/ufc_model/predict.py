"""Predict an upcoming card and append it to the hash-chained log.

``python -m ufc_model.predict --event "UFC 334" [--event-start 2026-11-14T22:00:00+00:00]
[--train] [--dry-run]``

Uses the latest registered production model (or trains and registers one with
``--train``). Only predictable bouts are logged; blocked bouts are listed with
their reasons. Without ``--event-start``, logging closes at 00:00 UTC on the
event date (conservative for cards anywhere in the world).
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import pandas as pd

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.logging_setup import setup_logging
from ufc_model.models import registry
from ufc_model.models.production import recent_holdout, train_production
from ufc_model.tracking import predictions_log

log = logging.getLogger("ufc_model.predict")


def train_and_register(cfg, rows: pd.DataFrame) -> tuple[object, dict]:
    holdout = recent_holdout(rows, cfg, cfg.production.holdout_days)
    model = train_production(rows, cfg)
    version, path = registry.save(
        model,
        cfg.production.registry_dir,
        {"recent_holdout": holdout},
        cfg.production.keep_versions,
    )
    log.info(
        "Registered %s (holdout: model %s, elo %s, T=%.3f%s)",
        version,
        holdout["model"],
        holdout["elo"],
        holdout["temperature"],
        " DRIFT FLAG" if holdout["temperature_drift_flag"] else "",
    )
    return registry.load_latest(cfg.production.registry_dir)


def build_records(
    model,
    meta: dict,
    card: pd.DataFrame,
    bouts: pd.DataFrame,
    event_name: str,
    event_start: str,
    top: int,
) -> list[dict]:
    pred = model.predict(card)
    contrib = model.contributions(card)
    info = bouts.set_index(bouts["wiki_title"] + "#" + bouts["bout_order"].astype(str))
    records = []
    for fid in sorted(pred.index, key=lambda f: int(f.split("#")[1])):
        r, b = pred.loc[fid], info.loc[fid]
        drivers = contrib.loc[fid].sort_values(key=abs, ascending=False).head(top)
        p_a = round(float(r["p"]), 4)
        records.append(
            {
                "event": event_name,
                "event_date": str(pd.Timestamp(r["event_date"]).date()),
                "event_start_utc": event_start,
                "bout_key": fid,
                "bout_order": int(b["bout_order"]),
                "weight_class": b["weight_class"],
                "scheduled_rounds": 5 if (b["is_title"] or b["is_main_event"]) else 3,
                "fighter_a": {"name": r["fighter_name"], "id": b["fighter_a_id"]},
                "fighter_b": {"name": r["opponent_name"], "id": b["fighter_b_id"]},
                "p_a": p_a,
                "p_b": round(1 - p_a, 4),
                "baselines": {"elo_p_a": round(float(r["elo_prob"]), 4)},
                "top_drivers": {k: round(float(v), 4) for k, v in drivers.items()},
                "model_version": meta["version"],
                "data_cutoff": meta["data_cutoff"],
                "source": {
                    "wiki_title": b["wiki_title"],
                    "wiki_revid": int(b["wiki_revid"]),
                    "card_retrieved_at": str(b["retrieved_at"]),
                },
            }
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="Wikipedia title, e.g. 'UFC 334'")
    ap.add_argument("--event-start", help="ISO time logging closes (default 00:00 UTC event day)")
    ap.add_argument("--train", action="store_true", help="Train and register a new model first")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = get_config()
    con = db.connect(cfg.paths.db_path)
    rows = con.sql("select * from fight_features order by event_date, fight_id, orientation").df()
    upcoming = con.sql("select * from upcoming_features").df()
    bouts = con.sql("select * from upcoming_bouts").df()
    events = con.sql("select * from upcoming_events").df()
    con.close()

    if args.train:
        model, meta = train_and_register(cfg, rows.reset_index(drop=True))
    else:
        model, meta = registry.load_latest(cfg.production.registry_dir)
    ev = events[events["wiki_title"] == args.event]
    if ev.empty:
        raise SystemExit(f"{args.event!r} is not among the loaded upcoming events")
    event_date = pd.Timestamp(ev["event_date"].iloc[0]).date()
    start = (
        args.event_start
        or datetime(event_date.year, event_date.month, event_date.day, tzinfo=UTC).isoformat()
    )
    card_bouts = bouts[bouts["wiki_title"] == args.event]
    for b in card_bouts[~card_bouts["predictable"]].itertuples():
        log.warning("BLOCKED %s vs %s: %s", b.fighter_a_name, b.fighter_b_name, b.block_reason)
    card = upcoming[upcoming["fight_id"].str.startswith(f"{args.event}#")]
    records = build_records(
        model, meta, card, card_bouts, ev["name"].iloc[0], start, cfg.production.top_drivers
    )
    if args.dry_run:
        for r in records:
            print(r["fighter_a"]["name"], r["p_a"], r["fighter_b"]["name"], r["p_b"])
        return
    written = predictions_log.append(cfg.production.predictions_log, records)
    log.info(
        "Logged %d predictions for %s (seq %d-%d), model %s",
        len(written),
        args.event,
        written[0]["seq"],
        written[-1]["seq"],
        meta["version"],
    )


if __name__ == "__main__":
    main()
