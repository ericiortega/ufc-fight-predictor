"""Parsers and the snapshot -> tables build, on a tiny synthetic snapshot."""

from datetime import date
from pathlib import Path

import pytest

from ufc_model.config import NamesConfig
from ufc_model.ingest.transform import (
    build_tables,
    fight_duration_s,
    parse_height_in,
    parse_inches,
    parse_mmss,
    parse_of,
    parse_outcome,
    parse_round_lengths,
    split_bout,
    split_location,
)
from ufc_model.weight_classes import parse_weight_class

NAMES_CFG = NamesConfig(
    suggestion_min_similarity=0.82, weight_margin_lbs=15, min_age_years=18, max_age_years=60
)


def test_field_parsers() -> None:
    assert parse_of("13 of 18") == (13, 18)
    assert parse_of(None) == (None, None)
    assert parse_of(float("nan")) == (None, None)
    assert parse_mmss("4:07") == 247
    assert parse_height_in("5' 11\"") == 71
    assert parse_inches('66"') == 66.0
    assert parse_inches(None) is None
    assert parse_outcome("W/L") == ("win", "A")
    assert parse_outcome("L/W") == ("win", "B")
    assert parse_outcome("NC/NC") == ("nc", None)
    assert split_location("Las Vegas, Nevada, USA") == ("Las Vegas", "Nevada", "USA")
    assert split_location("Abu Dhabi, United Arab Emirates") == (
        "Abu Dhabi",
        None,
        "United Arab Emirates",
    )
    assert split_bout("Renato Moicano vs. Tom Nolan") == ("Renato Moicano", "Tom Nolan")


@pytest.mark.parametrize(
    ("fmt", "lengths"),
    [("3 Rnd (5-5-5)", [5, 5, 5]), ("1 Rnd + OT (12-3)", [12, 3]), ("No Time Limit", None)],
)
def test_round_lengths(fmt, lengths) -> None:
    assert parse_round_lengths(fmt) == lengths


def test_fight_duration() -> None:
    assert fight_duration_s(3, 88, [5, 5, 5, 5, 5]) == 688
    assert fight_duration_s(1, 91, None) == 91
    assert fight_duration_s(2, 30, None) is None


@pytest.mark.parametrize(
    ("raw", "name", "lbs", "women", "title"),
    [
        ("Lightweight Bout", "Lightweight", 155, False, False),
        ("UFC Light Heavyweight Title Bout", "Light Heavyweight", 205, False, True),
        ("UFC Interim Women's Flyweight Title Bout", "Women's Flyweight", 125, True, True),
        (
            "Ultimate Fighter 28 Women's Featherweight Bout",
            "Women's Featherweight",
            145,
            True,
            False,
        ),
        ("Catch Weight Bout", "Catch Weight", None, False, False),
        ("Superfight Championship Bout", "Open Weight", None, False, True),
        ("Heavyweight", "Heavyweight", 265, False, False),
    ],
)
def test_weight_class(raw, name, lbs, women, title) -> None:
    wc = parse_weight_class(raw)
    assert (wc.name, wc.limit_lbs, wc.is_women, wc.is_title) == (name, lbs, women, title)


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


@pytest.fixture
def snapshot(tmp_path: Path) -> Path:
    """Two events; one fight duplicated under a renamed event; one name variant."""
    u = "http://ufcstats.com"
    _write(
        tmp_path / "ufc_event_details.csv",
        f"""
EVENT,URL,DATE,LOCATION
UFC 2: Late,{u}/event-details/e2,"February 01, 2024","Las Vegas, Nevada, USA"
UFC 1: Early,{u}/event-details/e1,"January 01, 2024","London, England, United Kingdom"
""",
    )
    _write(
        tmp_path / "ufc_fight_results.csv",
        f"""
EVENT,BOUT,OUTCOME,WEIGHTCLASS,METHOD,ROUND,TIME,TIME FORMAT,REFEREE,DETAILS,URL
UFC 2: Late ,Al Alpha vs. Zach Bravo,W/L,Lightweight Bout,KO/TKO ,2,1:30,3 Rnd (5-5-5),Ref,Punches ,{u}/fight-details/f2
UFC 1: Early ,Zach Bravo vs. Cy Charlie,L/W,Lightweight Bout,Decision - Unanimous ,3,5:00,3 Rnd (5-5-5),Ref,,{u}/fight-details/f1
Noche UFC 1: Early ,Zach Bravo vs. Cy Charlie,L/W,Lightweight Bout,Decision - Unanimous ,3,5:00,3 Rnd (5-5-5),Ref,,{u}/fight-details/f1
""",
    )
    _write(
        tmp_path / "ufc_fight_stats.csv",
        """
EVENT,BOUT,ROUND,FIGHTER,KD,SIG.STR.,SIG.STR. %,TOTAL STR.,TD,TD %,SUB.ATT,REV.,CTRL,HEAD,BODY,LEG,DISTANCE,CLINCH,GROUND
UFC 2: Late,Al Alpha vs. Zach Bravo,Round 1,Al Alpha,1,10 of 20,50%,12 of 22,1 of 2,50%,0,0,1:05,6 of 14,2 of 3,2 of 3,8 of 17,1 of 1,1 of 2
UFC 2: Late,Al Alpha vs. Zach Bravo,Round 1,Zach Bravo,0,5 of 15,33%,6 of 16,0 of 1,0%,1,0,0:00,3 of 10,1 of 3,1 of 2,5 of 15,0 of 0,0 of 0
UFC 1: Early,Zach Bravo vs. Cy Charlie,,,,,,,,,,,,,,,,,
""",
    )
    _write(
        tmp_path / "ufc_fighter_tott.csv",
        f'''
FIGHTER,HEIGHT,WEIGHT,REACH,STANCE,DOB,URL
Al Alpha,"5' 10""",155 lbs.,"70""",Orthodox,"Jan 01, 1995",{u}/fighter-details/a
Zachary Bravo,--,155 lbs.,--,,--,{u}/fighter-details/b
Cy Charlie,"6' 0""",155 lbs.,"74""",Southpaw,"Mar 03, 1990",{u}/fighter-details/c
''',
    )
    _write(
        tmp_path / "ufc_fighter_details.csv",
        f"""
FIRST,LAST,NICKNAME,URL
Al,Alpha,,{u}/fighter-details/a
Zach,Bravo,The Brave,{u}/fighter-details/b
Cy,Charlie,,{u}/fighter-details/c
""",
    )
    return tmp_path


def test_build_tables(snapshot: Path) -> None:
    res = build_tables(snapshot, {}, NAMES_CFG)
    ev, fighters, fights, rounds = (
        res.tables[k] for k in ("events", "fighters", "fights", "fight_rounds")
    )

    assert list(ev["event_id"]) == ["e2", "e1"]
    assert len(fights) == 2, "renamed-event duplicate must be dropped"
    assert not res.issues

    f2 = fights.set_index("fight_id").loc["f2"]
    assert (f2.fighter_a_id, f2.fighter_b_id, f2.winner_side) == ("a", "b", "A")
    assert f2.event_date == date(2024, 2, 1)
    assert f2.fight_duration_s == 390
    assert f2.has_round_stats
    f1 = fights.set_index("fight_id").loc["f1"]
    assert (f1.fighter_a_id, f1.winner_side, f1.has_round_stats) == ("b", "B", False)

    assert len(rounds) == 2
    alpha = rounds.set_index("side").loc["A"]
    assert (alpha.sig_str_landed, alpha.sig_str_attempted, alpha.ctrl_s, alpha.kd) == (
        10,
        20,
        65,
        1,
    )

    b = fighters.set_index("fighter_id").loc["b"]
    assert b.alt_name == "Zach Bravo" and b.nickname == "The Brave"
    assert fighters.set_index("fighter_id").loc["a"].height_in == 70


def test_limit_events_keeps_full_fighter_table(snapshot: Path) -> None:
    res = build_tables(snapshot, {}, NAMES_CFG, limit_events=1)
    assert list(res.tables["events"]["event_id"]) == ["e2"]
    assert list(res.tables["fights"]["fight_id"]) == ["f2"]
    assert len(res.tables["fighters"]) == 3
