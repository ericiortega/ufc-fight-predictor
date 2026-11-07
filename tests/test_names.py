"""Name matching must never silently attach a bout to the wrong fighter."""

from datetime import date

import pytest

from ufc_model.names import (
    Alias,
    Candidate,
    NameResolver,
    clean_display_name,
    compact_name,
    load_aliases,
    normalize_name,
)

FIGHTERS = [
    Candidate("uros", "Uros Medic", 170, date(1993, 1, 1)),
    Candidate("cortes", "Waldo Cortes-Acosta", 260, date(1991, 6, 1)),
    Candidate("yisak", "Yi Sak Lee", 185, date(2000, 1, 1)),
    Candidate("kamaka", "Kai Kamaka", 145, date(1995, 1, 1)),
    Candidate("bruno_flw", "Bruno Silva", 125, date(1990, 3, 16)),
    Candidate("bruno_mw", "Bruno Silva", 185, date(1989, 7, 13)),
    Candidate("mm_old", "Michael McDonald", 205, date(1965, 2, 6)),
    Candidate("mm_new", "Michael McDonald", 135, date(1991, 1, 15)),
    Candidate("romero_lw", "Anthony Romero", 155, date(1997, 2, 25)),
    Candidate("romero_fw", "Anthony Romero", 145, None),
    Candidate("ko", "Seokhyeon Ko", 170, date(1993, 9, 24)),
    Candidate("reese", "Zachary Reese", 185, None, alt_names=("Zach Reese",)),
    Candidate("freire", "Patricio Freire", 145, None, nickname="Pitbull"),
    Candidate("jacare", "Jacare Souza", 185, date(1979, 12, 7)),
]


@pytest.fixture
def resolver() -> NameResolver:
    return NameResolver(FIGHTERS)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Uroš Medić", "uros medic"),
        ("Waldo Cortes-Acosta", "waldo cortes acosta"),
        ("Kai Kamaka III", "kai kamaka"),
        ("Michael Aswell Jr.", "michael aswell"),
        ("Charles Oliveira (c)", "charles oliveira"),
        ("Sean O'Malley", "sean omalley"),
        ("  JIŘÍ   PROCHÁZKA ", "jiri prochazka"),
        ("Bøe Łukasz", "boe lukasz"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert normalize_name(raw) == expected


def test_clean_display_name_strips_markers() -> None:
    assert clean_display_name("Ciryl Gane (c) [12]") == "Ciryl Gane"
    assert compact_name("YiSak Lee") == compact_name("Yi Sak Lee")


def test_exact_match(resolver: NameResolver) -> None:
    m = resolver.resolve("Uros Medic")
    assert (m.fighter_id, m.method, m.confident, m.needs_review) == ("uros", "exact", True, False)


@pytest.mark.parametrize(
    ("raw", "fid", "method"),
    [
        ("Uroš Medić", "uros", "normalized"),
        ("Waldo Cortes Acosta", "cortes", "normalized"),
        ("Kai Kamaka III", "kamaka", "normalized"),
        ("YiSak Lee", "yisak", "compact"),
        ("Zach Reese", "reese", "alt_name"),  # ufcstats' own alternate name
    ],
)
def test_confident_non_exact_matches_are_logged(resolver, raw, fid, method) -> None:
    m = resolver.resolve(raw)
    assert (m.fighter_id, m.method, m.confident) == (fid, method, True)
    assert m.needs_review


def test_shared_name_resolved_by_weight(resolver: NameResolver) -> None:
    m = resolver.resolve("Bruno Silva", weight_class_lbs=125, fight_date=date(2023, 1, 1))
    assert (m.fighter_id, m.method) == ("bruno_flw", "tiebreak")
    m = resolver.resolve("Bruno Silva", weight_class_lbs=185, fight_date=date(2023, 1, 1))
    assert m.fighter_id == "bruno_mw"


def test_shared_name_resolved_by_age(resolver: NameResolver) -> None:
    m = resolver.resolve("Michael McDonald", fight_date=date(2012, 1, 1))  # old one would be 46
    assert m.fighter_id in {"mm_new", None}
    m = resolver.resolve("Michael McDonald", fight_date=date(1995, 1, 1))  # new one would be 3
    assert (m.fighter_id, m.method) == ("mm_old", "tiebreak")


def test_shared_name_inconclusive_is_unresolved(resolver: NameResolver) -> None:
    """10 lbs apart is inside the margin: refuse to guess."""
    m = resolver.resolve("Anthony Romero", weight_class_lbs=145, fight_date=date(2026, 10, 3))
    assert m.fighter_id is None and m.method == "unresolved" and not m.confident
    assert set(m.candidates) == {"romero_lw", "romero_fw"}


def test_shared_name_without_context_is_unresolved(resolver: NameResolver) -> None:
    assert resolver.resolve("Bruno Silva").fighter_id is None


@pytest.mark.parametrize(
    ("raw", "suggested"),
    [
        ("Ko Seok-hyeon", "ko"),  # reversed name order: must not become a "debut"
        ("Zachery Reese", "reese"),
        ("Patricio Pitbull", "freire"),  # first name + nickname
        ("Jose Henrique Souza", None),  # no confident match at all
    ],
)
def test_near_matches_are_never_auto_accepted(resolver, raw, suggested) -> None:
    m = resolver.resolve(raw, allow_new=True)
    if suggested is None:
        return
    assert m.fighter_id is None and m.method == "unresolved"
    assert any(f"[{suggested}]" in s for s in m.suggestions)


def test_new_fighter_only_when_nothing_is_close(resolver: NameResolver) -> None:
    m = resolver.resolve("Totally Unknown Debutant", allow_new=True)
    assert (m.fighter_id, m.method, m.confident) == (None, "new", True)
    m = resolver.resolve("Totally Unknown Debutant", allow_new=False)
    assert m.method == "unresolved"


@pytest.mark.parametrize("placeholder", ["TBD", "TBA", "To be determined", ""])
def test_placeholders_block(resolver: NameResolver, placeholder: str) -> None:
    m = resolver.resolve(placeholder, allow_new=True)
    assert m.fighter_id is None and m.method == "unresolved"


def test_implausible_age_blocks_single_candidate(resolver: NameResolver) -> None:
    m = resolver.resolve("Yi Sak Lee", fight_date=date(2010, 1, 1))  # would be 10
    assert m.fighter_id is None and m.method == "unresolved"


def test_aliases_and_scopes(tmp_path) -> None:
    path = tmp_path / "aliases.yaml"
    path.write_text(
        "aliases:\n"
        "  - {name: Ko Seok-hyeon, fighter_id: ko}\n"
        "  - {name: Anthony Romero, weight_class: Featherweight, fighter_id: romero_fw}\n"
        "  - {name: Jose Henrique Souza, fighter_id: new}\n"
    )
    aliases = load_aliases(path)
    r = NameResolver(FIGHTERS, aliases)
    assert r.resolve("Ko Seok-hyeon").fighter_id == "ko"
    assert r.resolve("Anthony Romero", weight_class_lbs=145).fighter_id == "romero_fw"
    # Out of scope: falls back to the normal (inconclusive) path.
    assert r.resolve("Anthony Romero", weight_class_lbs=155).method in {"tiebreak", "unresolved"}
    m = r.resolve("José Henrique Souza")
    assert (m.fighter_id, m.method) == (None, "new")


def test_alias_to_unknown_fighter_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fighter_ids"):
        NameResolver(FIGHTERS, {"someone": [Alias("does_not_exist")]})


def test_alias_unknown_weight_class_rejected(tmp_path) -> None:
    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  - {name: X Y, weight_class: Cruiserweight, fighter_id: ko}\n")
    with pytest.raises(ValueError, match="weight_class"):
        load_aliases(path)


def test_conflicting_upstream_alt_name_is_not_trusted() -> None:
    """Real upstream glitch: profile "Regina Malpica" carries the name "Tyrell Fortune"."""
    fighters = [
        Candidate(
            "malpica", "Regina Malpica", 125, date(2004, 10, 19), alt_names=("Tyrell Fortune",)
        ),
        Candidate("reese", "Zachary Reese", 185, None, alt_names=("Zach Reese",)),
    ]
    r = NameResolver(fighters)
    m = r.resolve("Tyrell Fortune", weight_class_lbs=265, allow_new=True)
    assert m.fighter_id is None and m.method == "unresolved"
    assert m.candidates == ["malpica"] and "different upstream profile" in m.note
    # Consistent variants are still used.
    assert r.resolve("Zach Reese").fighter_id == "reese"


@pytest.mark.parametrize(
    ("primary", "alt", "ok"),
    [
        ("Zachary Reese", "Zach Reese", True),
        ("Muhammad Said", "Muhammad Saidov", True),
        ("JonathanMix", "Jonathan Mix", True),
        ("Regina Malpica", "Tyrell Fortune", False),
        ("Jose Montanha", "Henrique Da Silva Lopes", False),
    ],
)
def test_alt_name_consistency(primary, alt, ok) -> None:
    from ufc_model.names import alt_name_consistent

    assert alt_name_consistent(primary, alt) is ok


def test_blocked_alias_overrides_a_match(tmp_path) -> None:
    path = tmp_path / "aliases.yaml"
    path.write_text("aliases:\n  - {name: Zach Reese, fighter_id: block}\n")
    fighters = [Candidate("reese", "Zachary Reese", 185, None, alt_names=("Zach Reese",))]
    m = NameResolver(fighters, load_aliases(path)).resolve("Zach Reese", allow_new=True)
    assert m.fighter_id is None and m.method == "unresolved" and "manual review" in m.note
