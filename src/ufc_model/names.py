"""Fighter name normalization and conservative identity resolution.

Policy: only *confident* matches resolve to a fighter. Anything ambiguous or
fuzzy is returned unresolved with logged suggestions, never guessed. Curated
aliases (``config/name_aliases.yaml``) are the only way to accept a fuzzy match.
"""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import yaml

log = logging.getLogger(__name__)

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
# Letters that NFKD does not decompose into ASCII + combining marks.
_CHAR_MAP = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "ł": "l",
        "Ł": "L",
        "đ": "d",
        "Đ": "D",
        "ß": "ss",
        "æ": "ae",
        "Æ": "Ae",
        "ı": "i",
        "ð": "d",
        "þ": "th",
    }
)

PLACEHOLDER_NAMES = {"tbd", "tba", "to be determined", "to be announced", "unknown", "opponent"}

MatchMethod = Literal["exact", "normalized", "compact", "alias", "tiebreak", "new", "unresolved"]


def clean_display_name(name: str) -> str:
    """Strip Wikipedia-style markers like "(c)", "(ic)" and "[12]" from a name."""
    name = re.sub(r"\[[^\]]*\]", "", name)
    name = re.sub(r"\((?:c|ic|uc)\)", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes: "Uroš Medić Jr." -> "uros medic"."""
    text = clean_display_name(name).translate(_CHAR_MAP)
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.lower().replace("-", " ")
    text = re.sub(r"[.'’`]", "", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = text.split()
    while len(tokens) > 1 and tokens[-1] in _SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def compact_name(name: str) -> str:
    """Normalized name with spaces removed: "YiSak Lee" == "Yi Sak Lee"."""
    return normalize_name(name).replace(" ", "")


@dataclass(frozen=True)
class Candidate:
    fighter_id: str
    name: str
    weight_lbs: float | None = None
    dob: date | None = None
    alt_names: tuple[str, ...] = ()
    nickname: str | None = None


@dataclass(frozen=True)
class Alias:
    """A curated mapping. ``fighter_id=None`` means "confirmed new/distinct fighter"."""

    fighter_id: str | None
    weight_class_lbs: int | None = None


@dataclass
class MatchResult:
    raw_name: str
    fighter_id: str | None
    method: MatchMethod
    confident: bool
    candidates: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def needs_review(self) -> bool:
        return self.method != "exact"


def load_aliases(path: Path) -> dict[str, list[Alias]]:
    """Load curated aliases as {normalized raw name: [Alias, ...]}.

    Each entry may be scoped to a ``weight_class`` (for names shared by two
    fighters). ``fighter_id: new`` marks a name confirmed to be a distinct
    fighter with no profile, silencing a misleading fuzzy suggestion.
    """
    from ufc_model.weight_classes import WEIGHT_CLASS_LBS

    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, list[Alias]] = defaultdict(list)
    for a in data.get("aliases") or []:
        wc = a.get("weight_class")
        if wc is not None and wc not in WEIGHT_CLASS_LBS:
            raise ValueError(f"Alias {a['name']!r}: unknown weight_class {wc!r}")
        fid = str(a["fighter_id"])
        out[normalize_name(a["name"])].append(
            Alias(None if fid == "new" else fid, WEIGHT_CLASS_LBS.get(wc) if wc else None)
        )
    return dict(out)


def _age_years(dob: date, on: date) -> float:
    return (on - dob).days / 365.25


class NameResolver:
    """Resolve raw fighter names from any source to ufcstats fighter IDs."""

    def __init__(
        self,
        fighters: list[Candidate],
        aliases: dict[str, list[Alias]] | None = None,
        *,
        weight_margin_lbs: float = 15,
        min_age_years: float = 18,
        max_age_years: float = 50,
        suggestion_min_similarity: float = 0.82,
    ) -> None:
        self.by_id = {c.fighter_id: c for c in fighters}
        self.aliases = aliases or {}
        self.weight_margin_lbs = weight_margin_lbs
        self.min_age = min_age_years
        self.max_age = max_age_years
        self.min_sim = suggestion_min_similarity
        self._exact: dict[str, list[Candidate]] = defaultdict(list)
        self._norm: dict[str, list[Candidate]] = defaultdict(list)
        self._compact: dict[str, list[Candidate]] = defaultdict(list)
        self._nick: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
        for c in fighters:
            for n in dict.fromkeys((c.name, *c.alt_names)):  # dedupe, keep order
                if c not in self._exact[n.strip()]:
                    self._exact[n.strip()].append(c)
                if c not in self._norm[normalize_name(n)]:
                    self._norm[normalize_name(n)].append(c)
                if c not in self._compact[compact_name(n)]:
                    self._compact[compact_name(n)].append(c)
            if c.nickname and (first := normalize_name(c.name).split()[:1]):
                self._nick[(first[0], normalize_name(c.nickname))].append(c)
        missing = [
            a.fighter_id
            for entries in self.aliases.values()
            for a in entries
            if a.fighter_id is not None and a.fighter_id not in self.by_id
        ]
        if missing:
            raise ValueError(f"Aliases point to unknown fighter_ids: {missing}")

    def _plausible_age(self, c: Candidate, on: date | None) -> bool:
        if c.dob is None or on is None:
            return True
        return self.min_age <= _age_years(c.dob, on) <= self.max_age

    def _suggest(self, raw: str) -> list[str]:
        """Near-matches for human review. Never auto-applied.

        Deliberately broad: a missed suggestion would let a known fighter be
        treated as a debut, which is worse than an extra name to review.
        """
        norm = normalize_name(raw)
        toks, comp = norm.split(), norm.replace(" ", "")
        sorted_comp = "".join(sorted(toks))
        out: list[tuple[float, str]] = []
        for key, cands in self._norm.items():
            ktoks, kcomp = key.split(), key.replace(" ", "")
            ratio = max(
                difflib.SequenceMatcher(None, norm, key).ratio(),
                difflib.SequenceMatcher(None, sorted_comp, "".join(sorted(ktoks))).ratio(),
            )
            multi = len(toks) > 1 and len(ktoks) > 1
            hit = (
                ratio >= self.min_sim
                # "Ko Seok-hyeon" vs "Seokhyeon Ko": every token of each inside the other
                or (all(t in kcomp for t in toks) and all(t in comp for t in ktoks))
                # "Zach Reese" vs "Zachary Reese": same surname, same first initial
                or (multi and toks[-1] == ktoks[-1] and toks[0][0] == ktoks[0][0])
                # "Jose Henrique Souza" vs "Jose Souza": first + last contained
                or (multi and toks[0] == ktoks[0] and toks[-1] in ktoks)
                or (multi and ktoks[0] == toks[0] and ktoks[-1] in toks)
            )
            if hit:
                out.extend((ratio, f"{c.name} [{c.fighter_id}]") for c in cands)
        # "Patricio Pitbull": first name + nickname
        if len(toks) > 1:
            out.extend(
                (1.0, f"{c.name} [{c.fighter_id}] (nickname)")
                for c in self._nick.get((toks[0], toks[-1]), [])
            )
        return list(dict.fromkeys(s for _, s in sorted(out, reverse=True)))[:5]

    def resolve(
        self,
        raw_name: str,
        *,
        weight_class_lbs: int | None = None,
        fight_date: date | None = None,
        allow_new: bool = False,
    ) -> MatchResult:
        """Resolve one name.

        ``allow_new``: if no candidate *and* no fuzzy suggestion exists, treat the
        name as a fighter absent from the dataset (e.g. a UFC debut) and return
        ``method="new"`` with ``fighter_id=None``. Otherwise such names are unresolved.
        """
        raw = clean_display_name(raw_name)
        norm = normalize_name(raw)
        if not norm or norm in PLACEHOLDER_NAMES:
            return MatchResult(
                raw, None, "unresolved", False, note="placeholder, opponent not named"
            )

        if norm in self.aliases:
            applicable = [
                a
                for a in self.aliases[norm]
                if a.weight_class_lbs is None or a.weight_class_lbs == weight_class_lbs
            ]
            if len(applicable) == 1:
                a = applicable[0]
                if a.fighter_id is None:
                    return MatchResult(raw, None, "new", True, note="curated: distinct fighter")
                return MatchResult(
                    raw, a.fighter_id, "alias", True, [a.fighter_id], note="curated alias"
                )

        method: MatchMethod
        for method, cands in (  # noqa: B007 - `method` is used after the loop
            ("exact", self._exact.get(raw, [])),
            ("normalized", self._norm.get(norm, [])),
            ("compact", self._compact.get(compact_name(raw), [])),
        ):
            if cands:
                break
        else:
            suggestions = self._suggest(raw)
            if allow_new and not suggestions:
                return MatchResult(raw, None, "new", True, note="no candidate or near-match")
            return MatchResult(
                raw,
                None,
                "unresolved",
                False,
                suggestions=suggestions,
                note="no confident candidate; see suggestions",
            )

        ids = [c.fighter_id for c in cands]
        plausible = [c for c in cands if self._plausible_age(c, fight_date)]
        if len(cands) == 1:
            if not plausible:
                return MatchResult(
                    raw,
                    None,
                    "unresolved",
                    False,
                    ids,
                    note=f"only candidate has implausible age on {fight_date}",
                )
            return MatchResult(raw, cands[0].fighter_id, method, True, ids)

        # Several fighters share this name: tie-break on age, then weight class.
        if len(plausible) == 1:
            return MatchResult(raw, plausible[0].fighter_id, "tiebreak", True, ids, note="age")
        if weight_class_lbs is not None and plausible:
            scored = sorted(
                (
                    (abs(c.weight_lbs - weight_class_lbs), c)
                    for c in plausible
                    if c.weight_lbs is not None
                ),
                key=lambda t: t[0],
            )
            if scored and (
                len(scored) == 1 or scored[1][0] - scored[0][0] >= self.weight_margin_lbs
            ):
                if len(scored) == len(plausible):
                    return MatchResult(
                        raw, scored[0][1].fighter_id, "tiebreak", True, ids, note="weight"
                    )
        return MatchResult(
            raw,
            None,
            "unresolved",
            False,
            ids,
            note=f"{len(cands)} fighters share this name; tie-breakers inconclusive",
        )
