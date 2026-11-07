"""Turn one upstream CSV snapshot into tidy tables: events, fighters, fights, fight_rounds.

Upstream fight rows identify fighters only by name, so every name goes through
:class:`~ufc_model.names.NameResolver`. Non-exact matches are logged for review.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from ufc_model.config import NamesConfig
from ufc_model.names import Alias, Candidate, MatchResult, NameResolver, compact_name
from ufc_model.weight_classes import parse_weight_class

log = logging.getLogger(__name__)

_MISSING = {"", "--", "---", "nan"}


# ---------------------------------------------------------------- field parsers


def _text(value: object) -> str:
    """Missing values arrive as None or NaN; treat any non-string as empty."""
    return value if isinstance(value, str) else ""


def _id_from_url(url: str) -> str:
    return url.rstrip("/").rsplit("/", 1)[-1]


def parse_of(value: str | None) -> tuple[int | None, int | None]:
    """ "13 of 18" -> (13, 18)."""
    m = re.fullmatch(r"\s*(\d+)\s+of\s+(\d+)\s*", _text(value))
    return (int(m[1]), int(m[2])) if m else (None, None)


def parse_mmss(value: str | None) -> int | None:
    """ "4:07" -> 247 seconds."""
    m = re.fullmatch(r"\s*(\d+):(\d{2})\s*", _text(value))
    return int(m[1]) * 60 + int(m[2]) if m else None


def parse_int(value: str | None) -> int | None:
    try:
        return int(float(value))  # tolerate "1.0"
    except (TypeError, ValueError):
        return None


def parse_height_in(value: str | None) -> int | None:
    """``5' 11"`` -> 71 inches."""
    m = re.fullmatch(r"\s*(\d+)'\s*(\d+)\"?\s*", _text(value))
    return int(m[1]) * 12 + int(m[2]) if m else None


def parse_inches(value: str | None) -> float | None:
    """``66"`` -> 66.0."""
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\"?\s*", _text(value))
    return float(m[1]) if m else None


def parse_lbs(value: str | None) -> float | None:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*lbs\.?\s*", _text(value))
    return float(m[1]) if m else None


def parse_date(value: str | None, fmt: str) -> date | None:
    try:
        return datetime.strptime(_text(value).strip(), fmt).date()
    except ValueError:
        return None


def parse_round_lengths(time_format: str | None) -> list[int] | None:
    """ "3 Rnd (5-5-5)" -> [5, 5, 5]; "1 Rnd + OT (12-3)" -> [12, 3]; else None."""
    m = re.search(r"\(([\d\-]+)\)", time_format or "")
    return [int(x) for x in m[1].split("-")] if m else None


def fight_duration_s(
    end_round: int | None, end_time_s: int | None, lengths: list[int] | None
) -> int | None:
    """Total elapsed fight time in seconds."""
    if end_round is None or end_time_s is None:
        return None
    if end_round == 1:
        return end_time_s
    if lengths is None or len(lengths) < end_round - 1:
        return None
    return sum(lengths[: end_round - 1]) * 60 + end_time_s


def parse_outcome(outcome: str | None) -> tuple[str | None, str | None]:
    """ "W/L" -> ("win", "A"), "L/W" -> ("win", "B"), draws/NCs have no winner side."""
    return {
        "W/L": ("win", "A"),
        "L/W": ("win", "B"),
        "D/D": ("draw", None),
        "NC/NC": ("nc", None),
    }.get(_text(outcome).strip(), (None, None))


def split_location(loc: str | None) -> tuple[str | None, str | None, str | None]:
    """ "Las Vegas, Nevada, USA" -> (city, region, country)."""
    parts = [p.strip() for p in _text(loc).split(",") if p.strip()]
    if not parts:
        return None, None, None
    if len(parts) == 1:
        return None, None, parts[0]
    return parts[0], (", ".join(parts[1:-1]) or None), parts[-1]


def split_bout(bout: str) -> tuple[str, str]:
    a, b = re.split(r"\s+vs\.?\s+", bout.strip(), maxsplit=1)
    return a.strip(), b.strip()


def read_csv(path: Path) -> pd.DataFrame:
    """Read every column as stripped text, with "--"/"" mapped to None."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].map(lambda v: None if v.strip() in _MISSING else v.strip())
    return df


# ---------------------------------------------------------------- build


@dataclass
class BuildResult:
    tables: dict[str, pd.DataFrame]
    match_log: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def build_fighters(snapshot: Path) -> pd.DataFrame:
    tott = read_csv(snapshot / "ufc_fighter_tott.csv")
    details = read_csv(snapshot / "ufc_fighter_details.csv")
    details["fighter_id"] = details["URL"].map(_id_from_url)
    details["alt_name"] = (
        details["FIRST"].fillna("") + " " + details["LAST"].fillna("")
    ).str.strip()
    tott["fighter_id"] = tott["URL"].map(_id_from_url)
    df = tott.merge(details[["fighter_id", "NICKNAME", "alt_name"]], on="fighter_id", how="left")
    alt = df["alt_name"].where(df["alt_name"].fillna("").ne("") & df["alt_name"].ne(df["FIGHTER"]))
    return pd.DataFrame(
        {
            "fighter_id": df["fighter_id"],
            "name": df["FIGHTER"],
            "alt_name": alt,
            "nickname": df["NICKNAME"],
            "height_in": df["HEIGHT"].map(parse_height_in),
            "weight_lbs": df["WEIGHT"].map(parse_lbs),
            "reach_in": df["REACH"].map(parse_inches),
            "stance": df["STANCE"],
            "dob": df["DOB"].map(lambda v: parse_date(v, "%b %d, %Y")),
            "is_provisional": False,
            "source_url": df["URL"],
        }
    )


def _opt(v: object) -> object | None:
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def make_resolver(
    fighters: pd.DataFrame, aliases: dict[str, list[Alias]], cfg: NamesConfig
) -> NameResolver:
    cands = [
        Candidate(
            r.fighter_id,
            r.name,
            _opt(r.weight_lbs),
            _opt(r.dob),
            alt_names=(r.alt_name,) if _opt(r.alt_name) else (),
            nickname=_opt(r.nickname),
        )
        for r in fighters.itertuples()
    ]
    return NameResolver(
        cands,
        aliases,
        weight_margin_lbs=cfg.weight_margin_lbs,
        min_age_years=cfg.min_age_years,
        max_age_years=cfg.max_age_years,
        suggestion_min_similarity=cfg.suggestion_min_similarity,
    )


def match_log_row(source: str, context: str, m: MatchResult) -> dict:
    return {
        "source": source,
        "context": context,
        "raw_name": m.raw_name,
        "fighter_id": m.fighter_id,
        "method": m.method,
        "confident": m.confident,
        "candidates": ", ".join(m.candidates),
        "suggestions": " | ".join(m.suggestions),
        "note": m.note,
    }


def build_tables(
    snapshot: Path,
    aliases: dict[str, list[Alias]],
    names_cfg: NamesConfig,
    limit_events: int | None = None,
) -> BuildResult:
    """Build all tidy tables from one snapshot directory.

    ``limit_events`` keeps only the N most recent events (for trial runs); the
    fighter dimension is always loaded in full so name resolution is unaffected.
    """
    result = BuildResult(tables={})
    issues, mlog = result.issues, result.match_log

    # --- events
    ev = read_csv(snapshot / "ufc_event_details.csv")
    city_region_country = ev["LOCATION"].map(split_location)
    events = pd.DataFrame(
        {
            "event_id": ev["URL"].map(_id_from_url),
            "name": ev["EVENT"],
            "event_date": ev["DATE"].map(lambda v: parse_date(v, "%B %d, %Y")),
            "location": ev["LOCATION"],
            "city": city_region_country.map(lambda t: t[0]),
            "region": city_region_country.map(lambda t: t[1]),
            "country": city_region_country.map(lambda t: t[2]),
            "source_url": ev["URL"],
        }
    )
    if events["event_date"].isna().any():
        issues.append(f"{events['event_date'].isna().sum()} events with unparseable dates")
    name_to_event = dict(zip(events["name"], events["event_id"], strict=True))

    # --- fight results: dedupe renamed-event duplicates, map to event_id
    res = read_csv(snapshot / "ufc_fight_results.csv").rename(
        columns={"TIME FORMAT": "TIME_FORMAT"}
    )
    res["_row"] = range(len(res))
    res["event_id"] = res["EVENT"].map(name_to_event)
    res["fight_id"] = res["URL"].map(_id_from_url)
    res = res.sort_values(["fight_id", "event_id"], na_position="last")
    n_dupes = res.duplicated("fight_id").sum()
    res = res.drop_duplicates("fight_id").sort_values("_row")
    if n_dupes:
        log.info("Dropped %d duplicate fight rows (events listed under two names)", n_dupes)
    unmapped = res[res["event_id"].isna()]
    for name, grp in unmapped.groupby("EVENT"):
        issues.append(f"Event '{name}' missing from event table; dropped {len(grp)} fights")
    res = res[res["event_id"].notna()].copy()
    res["card_position"] = res.groupby("event_id").cumcount() + 1

    if limit_events is not None:
        keep = (
            events[events["event_id"].isin(res["event_id"])]
            .sort_values("event_date", ascending=False)
            .head(limit_events)["event_id"]
        )
        events = events[events["event_id"].isin(keep)]
        res = res[res["event_id"].isin(keep)]
    event_date = dict(zip(events["event_id"], events["event_date"], strict=True))

    # --- fighters + identity resolution
    fighters = build_fighters(snapshot)
    resolver = make_resolver(fighters, aliases, names_cfg)
    for alt, cands in sorted(resolver.conflicting_alt_names.items()):
        profiles = ", ".join(f"{c.name} [{c.fighter_id}]" for c in cands)
        issues.append(f"Upstream alt name {alt!r} contradicts profile {profiles}; ignored")
    provisional: dict[str, str] = {}

    def resolve(raw: str, wc_lbs: int | None, on: date, fight_id: str) -> str | None:
        m = resolver.resolve(raw, weight_class_lbs=wc_lbs, fight_date=on, allow_new=True)
        if m.needs_review:
            mlog.append(match_log_row("ufcstats_fight", fight_id, m))
        if m.method == "new":
            pid = f"prov_{compact_name(raw)}"
            provisional.setdefault(pid, m.raw_name)
            return pid
        return m.fighter_id

    rows = []
    for r in res.itertuples():
        a_raw, b_raw = split_bout(r.BOUT)
        wc = parse_weight_class(r.WEIGHTCLASS)
        on = event_date[r.event_id]
        a_id = resolve(a_raw, wc.limit_lbs, on, r.fight_id)
        b_id = resolve(b_raw, wc.limit_lbs, on, r.fight_id)
        if a_id is not None and a_id == b_id:
            issues.append(f"Fight {r.fight_id}: both names resolved to {a_id}; nulled")
            a_id = b_id = None
        result_type, winner_side = parse_outcome(r.OUTCOME)
        end_round, end_time = parse_int(r.ROUND), parse_mmss(r.TIME)
        lengths = parse_round_lengths(r.TIME_FORMAT)
        rows.append(
            {
                "fight_id": r.fight_id,
                "event_id": r.event_id,
                "event_date": on,
                "card_position": r.card_position,
                "fighter_a_id": a_id,
                "fighter_b_id": b_id,
                "fighter_a_name": a_raw,
                "fighter_b_name": b_raw,
                "result": result_type,
                "winner_side": winner_side,
                "weight_class_raw": r.WEIGHTCLASS,
                "weight_class": wc.name,
                "weight_class_lbs": wc.limit_lbs,
                "is_title": wc.is_title,
                "is_women": wc.is_women,
                "method": r.METHOD,
                "method_detail": r.DETAILS,
                "end_round": end_round,
                "end_time_s": end_time,
                "time_format": r.TIME_FORMAT,
                "scheduled_rounds": len(lengths) if lengths else None,
                "fight_duration_s": fight_duration_s(end_round, end_time, lengths),
                "referee": r.REFEREE,
                "source_url": r.URL,
            }
        )
    fights = pd.DataFrame(rows)
    unresolved = fights["fighter_a_id"].isna().sum() + fights["fighter_b_id"].isna().sum()
    if unresolved:
        issues.append(f"{unresolved} fighter slots unresolved (see name_match_log)")
    bad_outcome = fights["result"].isna().sum()
    if bad_outcome:
        issues.append(f"{bad_outcome} fights with unknown outcome")

    if provisional:
        prov = pd.DataFrame(
            {
                "fighter_id": list(provisional),
                "name": list(provisional.values()),
                "is_provisional": True,
            }
        )
        fighters = pd.concat([fighters, prov], ignore_index=True)

    # --- round-by-round stats
    fights_by_key = res.set_index(["EVENT", "BOUT"])["fight_id"]
    dup_keys = set(fights_by_key.index[fights_by_key.index.duplicated()])
    if dup_keys:
        issues.append(f"Ambiguous (event, bout) keys, round stats skipped: {sorted(dup_keys)}")
    key_to_fight = {k: v for k, v in fights_by_key.items() if k not in dup_keys}
    sides = fights.set_index("fight_id")[
        ["fighter_a_name", "fighter_b_name", "fighter_a_id", "fighter_b_id"]
    ]

    st = read_csv(snapshot / "ufc_fight_stats.csv")
    st = st[st["ROUND"].notna() & st["FIGHTER"].notna()]
    st["fight_id"] = [key_to_fight.get(k) for k in zip(st["EVENT"], st["BOUT"], strict=True)]
    st = st[st["fight_id"].notna()]
    round_rows = []
    for r in st.to_dict("records"):
        s = sides.loc[r["fight_id"]]
        if r["FIGHTER"] == s.fighter_a_name:
            side, fid = "A", s.fighter_a_id
        elif r["FIGHTER"] == s.fighter_b_name:
            side, fid = "B", s.fighter_b_id
        else:
            issues.append(f"Fight {r['fight_id']}: stats row for unknown fighter {r['FIGHTER']!r}")
            continue
        row = {
            "fight_id": r["fight_id"],
            "side": side,
            "fighter_id": fid,
            "round": parse_int(r["ROUND"].replace("Round", "")),
            "kd": parse_int(r["KD"]),
        }
        for col, name in [
            ("SIG.STR.", "sig_str"),
            ("TOTAL STR.", "total_str"),
            ("TD", "td"),
            ("HEAD", "head"),
            ("BODY", "body"),
            ("LEG", "leg"),
            ("DISTANCE", "distance"),
            ("CLINCH", "clinch"),
            ("GROUND", "ground"),
        ]:
            row[f"{name}_landed"], row[f"{name}_attempted"] = parse_of(r[col])
        row["sub_att"] = parse_int(r["SUB.ATT"])
        row["reversals"] = parse_int(r["REV."])
        row["ctrl_s"] = parse_mmss(r["CTRL"])
        round_rows.append(row)
    rounds = pd.DataFrame(round_rows)
    if not rounds.empty:
        dupes = rounds.duplicated(["fight_id", "side", "round"]).sum()
        if dupes:
            issues.append(f"{dupes} duplicate round rows dropped")
            rounds = rounds.drop_duplicates(["fight_id", "side", "round"])
    fights["has_round_stats"] = fights["fight_id"].isin(
        rounds.get("fight_id", pd.Series(dtype=str))
    )

    result.tables = {
        "events": events,
        "fighters": fighters,
        "fights": fights,
        "fight_rounds": rounds,
    }
    return result
