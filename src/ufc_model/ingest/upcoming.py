"""Resolve Wikipedia upcoming cards to dataset fighter IDs, blocking unsafe bouts."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ufc_model.config import Config
from ufc_model.http import PoliteClient
from ufc_model.ingest.debut import find_debut_evidence
from ufc_model.ingest.transform import match_log_row
from ufc_model.ingest.wikipedia import CardBout, ScheduledEvent, fetch_page, parse_fight_card
from ufc_model.names import MatchResult, NameResolver
from ufc_model.weight_classes import parse_weight_class

log = logging.getLogger(__name__)

# Matched to an existing profile with confidence.
MATCHED_METHODS = {"exact", "alt_name", "normalized", "compact", "alias", "tiebreak"}
CURATED_NEW_NOTE = "curated: distinct fighter"


def _debut_status(m: MatchResult, html: str, others: list[str]) -> tuple[str | None, str | None]:
    """For a fighter with no profile match: (evidence, block reason)."""
    if m.note == CURATED_NEW_NOTE:
        return "curated alias: confirmed distinct fighter", None
    evidence = find_debut_evidence(html, m.raw_name, others)
    if evidence:
        return evidence, None
    return None, "no profile match and no debut evidence on the event page"


def resolve_card(
    bouts: list[CardBout], html: str, event_date: date, resolver: NameResolver, context: str
) -> tuple[list[dict], list[dict]]:
    """Resolve every bout on one card. Returns (bout rows, name-match log rows).

    A bout is predictable only if each fighter either matched a profile
    confidently, or has no profile *and* the event page says it is their UFC
    debut. A fighter with a near-match suggestion is never treated as a debut.
    """
    card_names = [n for b in bouts for n in (b.fighter_a_name, b.fighter_b_name)]
    rows, mlog = [], []
    for b in bouts:
        wc = parse_weight_class(b.weight_class_raw)
        reasons: list[str] = []
        sides = {}
        for side, name in (("a", b.fighter_a_name), ("b", b.fighter_b_name)):
            m = resolver.resolve(
                name, weight_class_lbs=wc.limit_lbs, fight_date=event_date, allow_new=True
            )
            if m.needs_review:
                mlog.append(match_log_row("wikipedia_card", f"{context}#{b.bout_order}", m))
            evidence = None
            if m.method == "new":
                others = [n for n in card_names if n != name]
                evidence, reason = _debut_status(m, html, others)
                if reason:
                    reasons.append(f"{side.upper()}: {m.raw_name!r} {reason}")
            elif m.method not in MATCHED_METHODS:
                reasons.append(f"{side.upper()}: {m.raw_name!r} {m.method} ({m.note})")
            sides[side] = (m, evidence)
        (a, a_ev), (bm, b_ev) = sides["a"], sides["b"]
        if a.fighter_id and a.fighter_id == bm.fighter_id:
            reasons.append("both names resolved to the same fighter")
        for r in reasons:
            log.warning("Bout blocked from prediction, %s: %s", context, r)
        rows.append(
            {
                "event_date": event_date,
                "bout_order": b.bout_order,
                "card_segment": b.card_segment,
                "weight_class_raw": b.weight_class_raw,
                "weight_class": wc.name,
                "fighter_a_name": b.fighter_a_name,
                "fighter_b_name": b.fighter_b_name,
                "fighter_a_id": a.fighter_id,
                "fighter_b_id": bm.fighter_id,
                "fighter_a_match": a.method,
                "fighter_b_match": bm.method,
                "fighter_a_debut_evidence": a_ev,
                "fighter_b_debut_evidence": b_ev,
                "predictable": not reasons,
                "block_reason": "; ".join(reasons) or None,
                "source_section": b.source_section,
            }
        )
    return rows, mlog


def build_upcoming(
    client: PoliteClient,
    cfg: Config,
    scheduled: list[ScheduledEvent],
    resolver: NameResolver,
    today: date,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Fetch the next N scheduled cards and resolve every fighter name."""
    upcoming = [e for e in scheduled if e.event_date >= today and e.wiki_title]
    upcoming = upcoming[: cfg.wikipedia.upcoming_events_to_fetch]
    event_rows, bout_rows, mlog = [], [], []
    for ev in upcoming:
        page = fetch_page(client, cfg, ev.wiki_title, cache_dir)
        event_rows.append(
            {
                "wiki_title": page.title,
                "name": ev.name,
                "event_date": ev.event_date,
                "venue": ev.venue,
                "location": ev.location,
                "wiki_revid": page.revid,
                "retrieved_at": page.retrieved_at,
            }
        )
        bouts = parse_fight_card(page.html)
        if not bouts:
            log.warning("No bouts parsed for %s (rev %s)", page.title, page.revid)
        rows, card_log = resolve_card(bouts, page.html, ev.event_date, resolver, page.title)
        for r in rows:
            r.update(wiki_title=page.title, wiki_revid=page.revid, retrieved_at=page.retrieved_at)
        bout_rows.extend(rows)
        mlog.extend(card_log)
    return pd.DataFrame(event_rows), pd.DataFrame(bout_rows), mlog
