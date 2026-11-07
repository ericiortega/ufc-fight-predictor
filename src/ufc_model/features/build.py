"""Chronological, point-in-time feature build.

Fights are processed one event *date* at a time:

1. For every fight on date D, snapshot both fighters from state that contains
   only fights before D (history, Elo, league priors, profile imputation stats).
2. Only then update the state with D's results.

Nothing from D or later can reach a feature for a fight on D. Each fight yields
two rows (A-vs-B and B-vs-A); difference features flip sign and the label flips,
which removes red/blue corner bias.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ufc_model.config import EloConfig, FeaturesConfig
from ufc_model.features.elo import EloParams, EloRatings, expected_score
from ufc_model.features.history import (
    FighterHistory,
    FightRecord,
    GlobalPriors,
    ShrinkConfig,
    Totals,
    method_category,
)
from ufc_model.features.profile import STANCES, Profile, ProfileImputer, stance_category

log = logging.getLogger(__name__)

PROFILE_FIELDS = ("height", "reach", "stance", "age")
_STANCE_FLAGS = [f"stance_{s}" for s in STANCES[:3]]

# Per-fighter values that become A-minus-B differences.
FIGHTER_KEYS: list[str] = [
    *FighterHistory().features(date(2000, 1, 1), GlobalPriors(), ShrinkConfig(1, 1, 1), 3),
    "elo",
    "height_in",
    "reach_in",
    "age_years",
    *_STANCE_FLAGS,
]
CONTEXT_NUMERIC = ["is_title", "is_women", "five_rounds"]
NUMERIC_FEATURES = [f"diff_{k}" for k in FIGHTER_KEYS] + CONTEXT_NUMERIC
CATEGORICAL_FEATURES = ["weight_class", "stance_matchup"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Symmetric columns: identical in both orientations of a fight.
SYMMETRIC_COLUMNS = CONTEXT_NUMERIC + CATEGORICAL_FEATURES


@dataclass
class FeatureState:
    """Everything the pipeline knows at a point in time."""

    histories: dict[str, FighterHistory]
    elo: EloRatings
    priors: GlobalPriors
    imputer: ProfileImputer
    shrink: ShrinkConfig
    recent_n: int
    last_fight: dict[str, date] = field(default_factory=dict)


def new_state(fcfg: FeaturesConfig, ecfg: EloConfig) -> FeatureState:
    return FeatureState(
        histories=defaultdict(FighterHistory),
        elo=EloRatings(EloParams(ecfg.initial, ecfg.k, ecfg.finish_multiplier)),
        priors=GlobalPriors(),
        imputer=ProfileImputer(
            fcfg.default_height_in,
            fcfg.default_reach_in,
            fcfg.default_age_years,
            fcfg.default_stance,
        ),
        shrink=ShrinkConfig(fcfg.shrink_minutes, fcfg.shrink_sig_attempts, fcfg.shrink_td_attempts),
        recent_n=fcfg.recent_fights,
    )


# ---------------------------------------------------------------- inputs


def load_profiles(fighters: pd.DataFrame) -> dict[str, Profile]:
    def opt(v):
        return None if v is None or pd.isna(v) else v

    return {
        r.fighter_id: Profile(
            height_in=opt(r.height_in) and float(r.height_in),
            reach_in=opt(r.reach_in) and float(r.reach_in),
            stance=stance_category(opt(r.stance)),
            dob=opt(r.dob) and pd.Timestamp(r.dob).date(),
        )
        for r in fighters.itertuples()
    }


def side_totals(rounds: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    """Sum round rows into one stat line per (fight_id, side)."""
    cols = {
        "sig_str_landed": "sig_l",
        "sig_str_attempted": "sig_a",
        "td_landed": "td_l",
        "td_attempted": "td_a",
        "sub_att": "sub_att",
        "ctrl_s": "ctrl_s",
        "kd": "kd",
    }
    agg = rounds.groupby(["fight_id", "side"])[list(cols)].sum(min_count=1).rename(columns=cols)
    return {k: v for k, v in agg.to_dict("index").items()}


def _fight_totals(stats, fight_id: str, side: str, duration_s) -> Totals | None:
    own, opp = stats.get((fight_id, side)), stats.get((fight_id, "B" if side == "A" else "A"))
    if own is None or opp is None or duration_s is None or pd.isna(duration_s) or duration_s <= 0:
        return None
    vals = [own[k] for k in own] + [opp[k] for k in opp]
    if any(pd.isna(v) for v in vals):
        return None
    return Totals(
        minutes=duration_s / 60,
        sig_l=own["sig_l"],
        sig_a=own["sig_a"],
        opp_sig_l=opp["sig_l"],
        opp_sig_a=opp["sig_a"],
        td_l=own["td_l"],
        td_a=own["td_a"],
        opp_td_l=opp["td_l"],
        opp_td_a=opp["td_a"],
        sub_att=own["sub_att"],
        ctrl_s=own["ctrl_s"],
        kd=own["kd"],
        opp_kd=opp["kd"],
    )


def _fid(value: object) -> str | None:
    """Fighter IDs arrive as str, or None/NaN when unresolved."""
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------- snapshots and rows


def snapshot(
    state: FeatureState,
    fighter_id: str | None,
    profile: Profile | None,
    weight_class: str | None,
    on: date,
) -> tuple[dict[str, float], dict[str, bool], str]:
    """Pre-fight features for one fighter, from state as of before ``on``."""
    hist = state.histories.get(fighter_id) if fighter_id else None
    feats = (hist or FighterHistory()).features(on, state.priors, state.shrink, state.recent_n)
    prof = state.imputer.impute(profile or Profile(None, None, None, None), weight_class, on)
    feats.update(
        elo=state.elo.get(fighter_id),
        height_in=prof.height_in,
        reach_in=prof.reach_in,
        age_years=prof.age_years,
        **{f"stance_{s}": float(prof.stance == s) for s in STANCES[:3]},
    )
    return feats, prof.imputed, prof.stance


def bout_rows(meta: dict, snap_a, snap_b) -> list[dict]:
    """Two mirrored rows for one bout. ``meta`` holds per-bout context and label."""
    (fa, ia, sa), (fb, ib, sb) = snap_a, snap_b
    context = {
        "is_title": float(bool(meta["is_title"])),
        "is_women": float(bool(meta["is_women"])),
        "five_rounds": float(meta.get("scheduled_rounds") == 5),
        "weight_class": meta["weight_class"]
        if isinstance(meta["weight_class"], str)
        else "Unknown",
        "stance_matchup": "-".join(sorted((sa, sb))),
    }
    label_a = meta.get("label_a")
    complete = not any(ia.values()) and not any(ib.values())
    rows = []
    for orient, (me, opp, imp_me, imp_opp, fid, oid, name, oname, label) in {
        "AB": (
            fa,
            fb,
            ia,
            ib,
            meta["fighter_a_id"],
            meta["fighter_b_id"],
            meta["fighter_a_name"],
            meta["fighter_b_name"],
            label_a,
        ),
        "BA": (
            fb,
            fa,
            ib,
            ia,
            meta["fighter_b_id"],
            meta["fighter_a_id"],
            meta["fighter_b_name"],
            meta["fighter_a_name"],
            None if label_a is None else 1 - label_a,
        ),
    }.items():
        row = {
            "fight_id": meta["fight_id"],
            "event_date": meta["event_date"],
            "orientation": orient,
            "fighter_id": fid,
            "opponent_id": oid,
            "fighter_name": name,
            "opponent_name": oname,
            "label": label,
            "result": meta.get("result"),
            "in_training": meta.get("in_training", False),
            "elo_prob": expected_score(me["elo"], opp["elo"]),
            "meta_profile_complete": complete,
            "meta_n_fights": me["n_fights"],
            "meta_opp_n_fights": opp["n_fights"],
            **{f"meta_missing_{f}": imp_me[f] for f in PROFILE_FIELDS},
            **{f"meta_opp_missing_{f}": imp_opp[f] for f in PROFILE_FIELDS},
            **{f"diff_{k}": me[k] - opp[k] for k in FIGHTER_KEYS},
            **context,
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------- the pass


@dataclass
class BuildOutput:
    rows: pd.DataFrame
    state: FeatureState
    notes: list[str]


def build_feature_table(
    fights: pd.DataFrame,
    rounds: pd.DataFrame,
    fighters: pd.DataFrame,
    fcfg: FeaturesConfig,
    ecfg: EloConfig,
) -> BuildOutput:
    """Build mirrored feature rows for every fight, strictly in date order."""
    state = new_state(fcfg, ecfg)
    profiles = load_profiles(fighters)
    stats = side_totals(rounds)
    fights = fights.sort_values(["event_date", "fight_id"])
    rows: list[dict] = []
    skipped_elo = unresolved = 0

    for on, day in fights.groupby("event_date", sort=True):
        on = pd.Timestamp(on).date()
        day_records = day.to_dict("records")
        # 1) snapshot every fight on this date from pre-date state
        for f in day_records:
            a, b = _fid(f["fighter_a_id"]), _fid(f["fighter_b_id"])
            label_a = {"A": 1, "B": 0}.get(f["winner_side"]) if f["result"] == "win" else None
            meta = {
                **f,
                "event_date": on,
                "label_a": label_a,
                "fighter_a_id": a,
                "fighter_b_id": b,
                "in_training": bool(
                    a and b and label_a is not None and on.year >= fcfg.train_start_year
                ),
            }
            snaps = [
                snapshot(
                    state,
                    meta[f"fighter_{s}_id"],
                    profiles.get(meta[f"fighter_{s}_id"]),
                    f["weight_class"],
                    on,
                )
                for s in "ab"
            ]
            rows.extend(bout_rows(meta, *snaps))
        # 2) only now apply this date's results
        for f in day_records:
            ids = {s: _fid(f[f"fighter_{s.lower()}_id"]) for s in "AB"}
            if ids["A"] is None or ids["B"] is None:
                unresolved += 1
            outcome = {
                "win": {
                    "A": "W" if f["winner_side"] == "A" else "L",
                    "B": "W" if f["winner_side"] == "B" else "L",
                },
                "draw": {"A": "D", "B": "D"},
            }.get(f["result"], {"A": "NC", "B": "NC"})
            method = method_category(f["method"])
            for side in "AB":
                tot = _fight_totals(stats, f["fight_id"], side, f["fight_duration_s"])
                if tot is not None:
                    state.priors.observe(tot)
                if ids[side] is not None:
                    state.histories[ids[side]].add(FightRecord(on, outcome[side], method, tot))
                    state.last_fight[ids[side]] = on
                state.imputer.observe(
                    ids[side],
                    profiles.get(ids[side]) or Profile(None, None, None, None),
                    f["weight_class"],
                    on,
                )
            if f["result"] in {"win", "draw"}:
                if ids["A"] is None and ids["B"] is None:
                    skipped_elo += 1
                else:
                    score = 0.5 if f["result"] == "draw" else float(f["winner_side"] == "A")
                    state.elo.update(ids["A"], ids["B"], score, method in {"ko", "sub"})

    notes = [
        f"{unresolved} fights with an unresolved fighter: resolved opponent's history and Elo "
        "updated (unresolved side at initial rating); excluded from training",
        f"{skipped_elo} fights skipped for Elo (both fighters unresolved)",
    ]
    for n in notes:
        log.info(n)
    df = pd.DataFrame(rows)
    nan_cols = [c for c in NUMERIC_FEATURES if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN in model features (never allowed): {nan_cols}")
    return BuildOutput(df, state, notes)


def upcoming_rows(state: FeatureState, bouts: pd.DataFrame, fighters: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for predictable upcoming bouts, from the final (current) state."""
    profiles = load_profiles(fighters)
    rows: list[dict] = []
    for b in bouts[bouts["predictable"]].to_dict("records"):
        on = pd.Timestamp(b["event_date"]).date()
        ids = {s: _fid(b[f"fighter_{s}_id"]) for s in "ab"}
        meta = {
            "fight_id": f"{b['wiki_title']}#{b['bout_order']}",
            "event_date": on,
            "fighter_a_id": ids["a"],
            "fighter_b_id": ids["b"],
            "fighter_a_name": b["fighter_a_name"],
            "fighter_b_name": b["fighter_b_name"],
            "weight_class": b["weight_class"],
            "is_title": bool(b["is_title"]),
            "is_women": str(b["weight_class"] or "").startswith("Women's"),
            "scheduled_rounds": 5 if b["is_title"] or b["is_main_event"] else 3,
        }
        snaps = [snapshot(state, ids[s], profiles.get(ids[s]), b["weight_class"], on) for s in "ab"]
        rows.extend(bout_rows(meta, *snaps))
    return pd.DataFrame(rows)
