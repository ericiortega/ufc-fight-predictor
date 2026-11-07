"""Ingest CLI: ``python -m ufc_model.ingest [--limit-events N] [--offline]``.

1. Look up the upstream commit and download a dated, SHA-pinned CSV snapshot.
2. Rebuild events / fighters / fights / fight_rounds from that snapshot.
3. Check upstream freshness against Wikipedia's list of past events (fails loudly).
4. Fetch upcoming cards from Wikipedia and resolve every fighter name.
5. Write everything to DuckDB, plus a name-match log and an ingest-run record.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, date, datetime

import pandas as pd

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.http import PoliteClient
from ufc_model.ingest import greco
from ufc_model.ingest.staleness import StaleDataError, check_staleness
from ufc_model.ingest.transform import build_tables, make_resolver
from ufc_model.ingest.upcoming import build_upcoming
from ufc_model.ingest.wikipedia import fetch_page, parse_events_list
from ufc_model.logging_setup import setup_logging
from ufc_model.names import load_aliases

log = logging.getLogger("ufc_model.ingest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit-events",
        type=int,
        default=None,
        help="Load only the N most recent events (trial runs).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the latest local snapshot; skip all network calls.",
    )
    parser.add_argument("--skip-upcoming", action="store_true")
    args = parser.parse_args(argv)

    setup_logging()
    cfg = get_config()
    started = datetime.now(UTC)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    today = date.today()
    client = PoliteClient(cfg.http, cfg.project.user_agent, seed=cfg.project.seed)
    log.info("Run %s | %s", run_id, greco.SOURCE_CREDIT)

    # 1. Snapshot
    if args.offline:
        snap = greco.latest_local_snapshot(cfg)
    else:
        commit = greco.latest_upstream_commit(client, cfg)
        log.info("Upstream head %s (%s): %s", commit.sha[:10], commit.committed_at, commit.message)
        snap = greco.download_snapshot(client, cfg, commit)
    greco.verify_snapshot(snap)
    manifest = greco.read_manifest(snap)

    # 2. Tables
    aliases = load_aliases(cfg.paths.aliases)
    built = build_tables(snap, aliases, cfg.names, limit_events=args.limit_events)
    t = built.tables
    for issue in built.issues:
        log.warning("Data issue: %s", issue)

    # 3. Freshness (always against the *full* event table, even on trial runs)
    staleness_status = "skipped (offline)"
    scheduled = []
    wiki_cache = cfg.paths.raw_dir / "wikipedia" / today.isoformat()
    if not args.offline:
        listing = fetch_page(client, cfg, cfg.wikipedia.events_list_page, wiki_cache)
        scheduled, past_dates = parse_events_list(listing.html)
        all_events = pd.read_csv(snap / "ufc_event_details.csv")
        dataset_latest = (
            pd.to_datetime(all_events["DATE"].str.strip(), format="%B %d, %Y").max().date()
        )
        try:
            report = check_staleness(dataset_latest, past_dates, today, cfg.staleness.grace_days)
        except StaleDataError:
            log.exception("STALE UPSTREAM DATA - failing run")
            raise
        staleness_status = report.status
        log.info(
            "Freshness: %s (dataset ends %s, last completed event %s)",
            report.status,
            report.dataset_latest,
            report.last_completed_event,
        )

    # 4. Upcoming cards (resolved against the full fighter table)
    up_events, up_bouts = pd.DataFrame(), pd.DataFrame()
    if not args.offline and not args.skip_upcoming:
        resolver = make_resolver(t["fighters"], aliases, cfg.names)
        up_events, up_bouts, up_log = build_upcoming(
            client, cfg, scheduled, resolver, today, wiki_cache
        )
        built.match_log.extend(up_log)

    # 5. Persist
    con = db.connect(cfg.paths.db_path)
    try:
        con.execute("BEGIN")
        for name in db.SNAPSHOT_TABLES:
            db.replace_table(con, name, t[name])
        if not args.offline and not args.skip_upcoming:
            db.replace_table(con, "upcoming_events", up_events)
            db.replace_table(con, "upcoming_bouts", up_bouts)
        mlog = pd.DataFrame(built.match_log)
        if not mlog.empty:
            mlog.insert(0, "run_id", run_id)
        db.append_rows(con, "name_match_log", mlog)
        db.append_rows(
            con,
            "ingest_runs",
            pd.DataFrame(
                [
                    {
                        "run_id": run_id,
                        "started_at": started,
                        "finished_at": datetime.now(UTC),
                        "upstream_sha": manifest.upstream_sha,
                        "upstream_committed_at": manifest.upstream_committed_at,
                        "snapshot": snap.name,
                        "limit_events": args.limit_events,
                        "n_events": len(t["events"]),
                        "n_fights": len(t["fights"]),
                        "n_rounds": len(t["fight_rounds"]),
                        "n_fighters": len(t["fighters"]),
                        "n_upcoming_bouts": len(up_bouts),
                        "staleness_status": staleness_status,
                        "issues": " | ".join(built.issues) or None,
                    }
                ]
            ),
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    blocked = int((~up_bouts["predictable"]).sum()) if not up_bouts.empty else 0
    log.info(
        "Done: %d events, %d fights, %d round rows, %d fighters, %d upcoming bouts (%d blocked), "
        "%d name matches logged for review",
        len(t["events"]),
        len(t["fights"]),
        len(t["fight_rounds"]),
        len(t["fighters"]),
        len(up_bouts),
        blocked,
        len(built.match_log),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
