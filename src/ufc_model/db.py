"""DuckDB schema and loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY, name VARCHAR, event_date DATE, location VARCHAR,
    city VARCHAR, region VARCHAR, country VARCHAR, source_url VARCHAR
);
CREATE TABLE IF NOT EXISTS fighters (
    fighter_id VARCHAR PRIMARY KEY, name VARCHAR, alt_name VARCHAR, nickname VARCHAR,
    height_in INTEGER,
    weight_lbs DOUBLE, reach_in DOUBLE, stance VARCHAR, dob DATE,
    is_provisional BOOLEAN, source_url VARCHAR
);
CREATE TABLE IF NOT EXISTS fights (
    fight_id VARCHAR PRIMARY KEY, event_id VARCHAR, event_date DATE, card_position INTEGER,
    fighter_a_id VARCHAR, fighter_b_id VARCHAR, fighter_a_name VARCHAR, fighter_b_name VARCHAR,
    result VARCHAR, winner_side VARCHAR, weight_class_raw VARCHAR, weight_class VARCHAR,
    weight_class_lbs INTEGER, is_title BOOLEAN, is_women BOOLEAN, method VARCHAR,
    method_detail VARCHAR, end_round INTEGER, end_time_s INTEGER, time_format VARCHAR,
    scheduled_rounds INTEGER, fight_duration_s INTEGER, referee VARCHAR, source_url VARCHAR,
    has_round_stats BOOLEAN
);
CREATE TABLE IF NOT EXISTS fight_rounds (
    fight_id VARCHAR, side VARCHAR, fighter_id VARCHAR, round INTEGER, kd INTEGER,
    sig_str_landed INTEGER, sig_str_attempted INTEGER,
    total_str_landed INTEGER, total_str_attempted INTEGER,
    td_landed INTEGER, td_attempted INTEGER,
    head_landed INTEGER, head_attempted INTEGER, body_landed INTEGER, body_attempted INTEGER,
    leg_landed INTEGER, leg_attempted INTEGER, distance_landed INTEGER, distance_attempted INTEGER,
    clinch_landed INTEGER, clinch_attempted INTEGER,
    ground_landed INTEGER, ground_attempted INTEGER,
    sub_att INTEGER, reversals INTEGER, ctrl_s INTEGER,
    PRIMARY KEY (fight_id, side, round)
);
CREATE TABLE IF NOT EXISTS upcoming_events (
    wiki_title VARCHAR, name VARCHAR, event_date DATE, venue VARCHAR, location VARCHAR,
    wiki_revid BIGINT, retrieved_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS upcoming_bouts (
    wiki_title VARCHAR, event_date DATE, bout_order INTEGER, card_segment VARCHAR,
    is_title BOOLEAN, is_main_event BOOLEAN,
    weight_class_raw VARCHAR, weight_class VARCHAR,
    fighter_a_name VARCHAR, fighter_b_name VARCHAR,
    fighter_a_id VARCHAR, fighter_b_id VARCHAR,
    fighter_a_match VARCHAR, fighter_b_match VARCHAR,
    fighter_a_debut_evidence VARCHAR, fighter_b_debut_evidence VARCHAR,
    predictable BOOLEAN, block_reason VARCHAR, source_section VARCHAR,
    wiki_revid BIGINT, retrieved_at TIMESTAMPTZ
);
CREATE TABLE IF NOT EXISTS name_match_log (
    run_id VARCHAR, source VARCHAR, context VARCHAR, raw_name VARCHAR, fighter_id VARCHAR,
    method VARCHAR, confident BOOLEAN, candidates VARCHAR, suggestions VARCHAR, note VARCHAR
);
CREATE TABLE IF NOT EXISTS ingest_runs (
    run_id VARCHAR PRIMARY KEY, started_at TIMESTAMPTZ, finished_at TIMESTAMPTZ,
    upstream_sha VARCHAR, upstream_committed_at VARCHAR, snapshot VARCHAR,
    limit_events INTEGER, n_events INTEGER, n_fights INTEGER, n_rounds INTEGER,
    n_fighters INTEGER, n_upcoming_bouts INTEGER, staleness_status VARCHAR, issues VARCHAR
);
"""

# Tables fully rebuilt from each snapshot (deterministic and reproducible).
SNAPSHOT_TABLES = ["events", "fighters", "fights", "fight_rounds"]
UPCOMING_TABLES = ["upcoming_events", "upcoming_bouts"]


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


# DDL per table, so rebuilt tables always get the current schema.
TABLE_DDL = {
    stmt.split("EXISTS", 1)[1].split("(", 1)[0].strip(): stmt.strip() + ";"
    for stmt in SCHEMA.split(";")
    if stmt.strip()
}


def replace_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    """Recreate ``name`` from its current DDL and fill it with ``df`` (columns by name)."""
    con.execute(f"DROP TABLE IF EXISTS {name}")
    con.execute(TABLE_DDL[name])
    if df.empty:
        return
    con.register("_df", df)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _df")
    con.unregister("_df")


def append_rows(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    con.register("_df", df)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    con.execute(f"INSERT INTO {name} ({cols}) SELECT {cols} FROM _df")
    con.unregister("_df")
