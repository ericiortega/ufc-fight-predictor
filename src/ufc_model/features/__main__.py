"""Build features: ``python -m ufc_model.features``.

Writes to DuckDB:
- ``fight_features``: two mirrored rows per historical fight
- ``upcoming_features``: two rows per predictable upcoming bout
- ``elo_current``: every resolved fighter's rating after the last loaded event
"""

from __future__ import annotations

import logging

import pandas as pd

from ufc_model import db
from ufc_model.config import get_config
from ufc_model.features.build import build_feature_table, upcoming_rows
from ufc_model.logging_setup import setup_logging

log = logging.getLogger("ufc_model.features")


def main() -> None:
    setup_logging()
    cfg = get_config()
    con = db.connect(cfg.paths.db_path)
    try:
        fights = con.sql("select * from fights").df()
        rounds = con.sql("select * from fight_rounds").df()
        fighters = con.sql("select * from fighters").df()
        bouts = con.sql("select * from upcoming_bouts").df()

        out = build_feature_table(fights, rounds, fighters, cfg.features, cfg.elo)
        upcoming = upcoming_rows(out.state, bouts, fighters) if not bouts.empty else pd.DataFrame()
        names = dict(zip(fighters["fighter_id"], fighters["name"], strict=True))
        elo = pd.DataFrame(
            [
                {
                    "fighter_id": fid,
                    "name": names.get(fid),
                    "elo": r,
                    "n_fights": len(out.state.histories[fid].records),
                    "last_fight": out.state.last_fight.get(fid),
                }
                for fid, r in out.state.elo.ratings.items()
            ]
        ).sort_values("elo", ascending=False)

        con.execute("BEGIN")
        for name, df in (
            ("fight_features", out.rows),
            ("upcoming_features", upcoming),
            ("elo_current", elo),
        ):
            con.execute(f"DROP TABLE IF EXISTS {name}")
            if not df.empty:
                con.register("_df", df)
                con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
                con.unregister("_df")
        con.execute("COMMIT")
    finally:
        con.close()

    train = out.rows[out.rows["in_training"]]
    log.info(
        "fight_features: %d rows (%d fights); training rows %d (%d fights, %s+); upcoming rows %d",
        len(out.rows),
        out.rows["fight_id"].nunique(),
        len(train),
        train["fight_id"].nunique(),
        cfg.features.train_start_year,
        len(upcoming),
    )


if __name__ == "__main__":
    main()
