"""Positive evidence that a fighter is making their UFC debut, from an event page.

A fighter with no profile match is only treated as a debut when the event page
explicitly says so ("promotional newcomer X", "X will make his promotional
debut"). Without such a sentence the bout is blocked rather than assumed.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from ufc_model.names import normalize_name

# "promotional debut" counts; "heavyweight debut" (a weight-class move) does not.
_DEBUT_QUALIFIERS = {"promotional", "ufc", "octagon", "organizational"}
_NEWCOMER = {"newcomer", "newcomers"}
_LOOKAHEAD_TOKENS = 15


def _sentences(html: str) -> list[str]:
    """Prose sentences from the page body, excluding references and navigation."""
    soup = BeautifulSoup(html, "lxml")
    for junk in soup.select("ol.references, div.reflist, .navbox, sup, style, .mw-editsection"):
        junk.decompose()
    text = " ".join(el.get_text(" ", strip=True) for el in soup.find_all(["p", "li"]))
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _find(tokens: list[str], name: list[str], start: int = 0) -> list[int]:
    n = len(name)
    return [i for i in range(start, len(tokens) - n + 1) if tokens[i : i + n] == name]


def find_debut_evidence(html: str, fighter_name: str, other_names: list[str]) -> str | None:
    """Return the sentence that marks ``fighter_name`` as a UFC debutant, if any.

    ``other_names`` are the other fighters on the card; a debut phrase only
    counts if no other fighter is named between this fighter and the phrase.
    """
    target = normalize_name(fighter_name).split()
    others = [
        normalize_name(n).split() for n in other_names if normalize_name(n) != " ".join(target)
    ]
    if not target:
        return None
    for sentence in _sentences(html):
        tokens = normalize_name(sentence).split()
        for i in _find(tokens, target):
            # "promotional newcomer X" / "newcomers A and X"
            before = tokens[max(0, i - 6) : i]
            if before and before[-1] in _NEWCOMER:
                return sentence
            if len(before) >= 3 and before[-1] == "and":
                marks = [k for k, t in enumerate(before) if t in _NEWCOMER]
                if marks and before[marks[-1] + 1 : -1] in others:
                    return sentence
            # "X ... will make his promotional debut" with no other fighter in between
            after = tokens[i + len(target) : i + len(target) + _LOOKAHEAD_TOKENS]
            for j, tok in enumerate(after):
                if tok in {"debut", "debuts"} and j > 0 and after[j - 1] in _DEBUT_QUALIFIERS:
                    if not any(_find(after[:j], o) for o in others):
                        return sentence
                    break
    return None
