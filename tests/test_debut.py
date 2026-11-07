"""A fighter is treated as a UFC debut only on positive evidence; otherwise the bout is blocked."""

from datetime import date

import pytest

from ufc_model.ingest.debut import find_debut_evidence
from ufc_model.ingest.upcoming import resolve_card
from ufc_model.ingest.wikipedia import CardBout
from ufc_model.names import Alias, Candidate, NameResolver

FIGHTERS = [
    Candidate("ko", "Seokhyeon Ko", 170, date(1993, 9, 24)),
    Candidate("turman", "Wellington Turman", 170, date(1996, 6, 1)),
    Candidate("hines", "Alvin Hines", 265, date(1990, 1, 1)),
    Candidate("matthews", "Jake Matthews", 170, date(1994, 8, 19)),
]
EVENT_DATE = date(2026, 11, 7)


def _page(*paragraphs: str) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    refs = '<ol class="references"><li>"Newcomer Ko Seok-hyeon steps in" . ESPN.</li></ol>'
    return f"<div>{body}</div>{refs}"


def _card(*pairs: tuple[str, str]) -> list[CardBout]:
    return [
        CardBout(i + 1, "Main card", "Welterweight", a, b, "fight_card")
        for i, (a, b) in enumerate(pairs)
    ]


def _resolve(bouts, html, aliases=None):
    rows, log = resolve_card(bouts, html, EVENT_DATE, NameResolver(FIGHTERS, aliases), "Test Event")
    return rows, log


# ---------------------------------------------------------------- the Ko Seok-hyeon case


def test_ko_seok_hyeon_is_blocked_not_assumed_debut() -> None:
    """Reversed name order, no alias: a near-match exists, so the bout is blocked."""
    rows, log = _resolve(
        _card(("Ko Seok-hyeon", "Wellington Turman")), _page("A welterweight bout.")
    )
    (row,) = rows
    assert row["predictable"] is False
    assert row["fighter_a_id"] is None and row["fighter_a_match"] == "unresolved"
    assert "Seokhyeon Ko [ko]" in log[0]["suggestions"]


def test_ko_seok_hyeon_blocked_even_with_debut_wording() -> None:
    """Debut evidence never overrides a near-match: he is an existing fighter."""
    html = _page("Promotional newcomer Ko Seok-hyeon will face Wellington Turman.")
    (row,), _ = _resolve(_card(("Ko Seok-hyeon", "Wellington Turman")), html)
    assert row["predictable"] is False
    assert row["fighter_a_debut_evidence"] is None


def test_ko_seok_hyeon_with_alias_is_predictable() -> None:
    aliases = {"ko seok hyeon": [Alias("ko")]}
    (row,), _ = _resolve(_card(("Ko Seok-hyeon", "Wellington Turman")), _page(""), aliases)
    assert row["predictable"] is True
    assert (row["fighter_a_id"], row["fighter_a_match"]) == ("ko", "alias")


# ---------------------------------------------------------------- true debutants


def test_debut_with_evidence_is_predictable() -> None:
    html = _page(
        "A heavyweight bout between Alvin Hines and promotional newcomer Gabriel Lorenzo was scheduled."
    )
    (row,), _ = _resolve(_card(("Alvin Hines", "Gabriel Lorenzo")), html)
    assert row["predictable"] is True
    assert row["fighter_b_match"] == "new"
    assert "promotional newcomer Gabriel Lorenzo" in row["fighter_b_debut_evidence"]


def test_debut_without_evidence_is_blocked() -> None:
    (row,), _ = _resolve(_card(("Alvin Hines", "Gabriel Lorenzo")), _page("A heavyweight bout."))
    assert row["predictable"] is False
    assert "no debut evidence" in row["block_reason"]


def test_curated_new_alias_counts_as_evidence() -> None:
    aliases = {"gabriel lorenzo": [Alias(None)]}
    (row,), _ = _resolve(_card(("Alvin Hines", "Gabriel Lorenzo")), _page(""), aliases)
    assert row["predictable"] is True


# ---------------------------------------------------------------- evidence parsing


OTHERS = ["Jake Matthews", "Alvin Hines", "Wellington Turman"]


@pytest.mark.parametrize(
    "sentence",
    [
        "A welterweight bout between Jake Matthews and promotional newcomer Akbar Abdullaev was announced.",
        "Akbar Abdullaev is expected to make his promotional debut against Jake Matthews.",
        "Akbar Abdullaev, the Contender Series winner, will make his UFC debut.",
        "Promotional newcomers Alvin Hines and Akbar Abdullaev are expected to meet.",
        "Brennan withdrew and was replaced by promotional newcomer Akbar Abdullaev on four days' notice.",
    ],
)
def test_debut_evidence_found(sentence: str) -> None:
    assert find_debut_evidence(_page(sentence), "Akbar Abdullaev", OTHERS) == sentence


@pytest.mark.parametrize(
    "sentence",
    [
        # Debut belongs to another fighter named in between.
        "Akbar Abdullaev will face Jake Matthews, who makes his promotional debut.",
        # A weight-class debut is not a UFC debut.
        "Akbar Abdullaev will make his heavyweight debut.",
        # Not about this fighter at all.
        "Jake Matthews faces promotional newcomer Wellington Turman.",
        "Akbar Abdullaev returns after a long layoff.",
    ],
)
def test_debut_evidence_not_found(sentence: str) -> None:
    assert find_debut_evidence(_page(sentence), "Akbar Abdullaev", OTHERS) is None


def test_reference_titles_are_not_evidence() -> None:
    """The citation list mentions 'Newcomer Ko Seok-hyeon'; it must be ignored."""
    assert find_debut_evidence(_page("Nothing here."), "Ko Seok-hyeon", OTHERS) is None
