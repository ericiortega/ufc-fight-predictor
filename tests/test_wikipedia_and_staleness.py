"""Wikipedia parsing (on HTML shaped like the real MediaWiki output) and freshness checks."""

from datetime import date

import pytest

from ufc_model.ingest.staleness import StaleDataError, check_staleness
from ufc_model.ingest.wikipedia import parse_events_list, parse_fight_card

EVENTS_LIST = """
<table id="Scheduled_events"><tbody>
<tr><th>Event</th><th>Date</th><th>Venue</th><th>Location</th><th>Ref.</th></tr>
<tr><td><a href="/wiki/UFC_335">UFC 335: Oliveira vs. Lopes</a></td><td>Dec 12, 2026</td>
    <td>T-Mobile Arena</td><td>Las Vegas , Nevada , U.S.</td><td><sup>[17]</sup></td></tr>
<tr><td><a href="/wiki/UFC_Fight_Night:_Proch%C3%A1zka_vs._Stirling">UFC Fight Night: Procházka vs. Stirling</a></td>
    <td>Nov 21, 2026</td><td>TBA</td><td>TBA</td><td></td></tr>
</tbody></table>
<table id="Past_events"><tbody>
<tr><th>#</th><th>Event</th><th>Date</th><th>Venue</th></tr>
<tr><td>796</td><td>UFC Fight Night: Moicano vs. Nolan</td><td>Oct 31, 2026</td><td>Meta Apex</td></tr>
<tr><td>795</td><td>UFC 333</td><td>Oct 24, 2026</td><td>Etihad Arena</td></tr>
</tbody></table>
"""

CARD = """
<table class="toccolours"><tbody>
<tr><th colspan="8">Main card (Paramount+)</th></tr>
<tr><th>Weight class</th><th></th><th></th><th></th><th>Method</th><th>Round</th><th>Time</th><th>Notes</th></tr>
<tr><td>Heavyweight</td><td><a>Ciryl Gane</a> (c)</td><td>vs.</td><td>Josh Hokit</td><td></td><td></td><td></td><td><sup>[a]</sup></td></tr>
<tr><td>Women's Bantamweight</td><td>Kayla Harrison (c)</td><td>vs.</td><td>Amanda Nunes</td><td></td><td></td><td></td><td></td></tr>
<tr><th colspan="8">Preliminary card (Paramount+)</th></tr>
<tr><td>Welterweight</td><td>Uroš Medić</td><td>vs.</td><td>Kevin Holland</td><td></td><td></td><td></td><td></td></tr>
</tbody></table>
<div class="mw-heading mw-heading2"><h2 id="Announced_bouts">Announced bouts</h2></div>
<ul>
<li>Bantamweight bout: Juan Díaz vs. Alateng Heili <sup>[17]</sup></li>
<li>Welterweight Bout: Uroš Medić vs. Kevin Holland</li>
</ul>
<div class="mw-heading mw-heading2"><h2 id="See_also">See also</h2></div>
<ul><li>Lightweight bout: Not A vs. Real Bout</li></ul>
"""


def test_parse_events_list() -> None:
    scheduled, past = parse_events_list(EVENTS_LIST)
    assert [e.event_date for e in scheduled] == [date(2026, 11, 21), date(2026, 12, 12)]
    assert scheduled[0].wiki_title == "UFC Fight Night: Procházka vs. Stirling"
    assert scheduled[1].location == "Las Vegas, Nevada, U.S."
    assert past == [date(2026, 10, 31), date(2026, 10, 24)]


def test_parse_events_list_fails_on_layout_change() -> None:
    with pytest.raises(ValueError, match="Scheduled_events"):
        parse_events_list("<html></html>")


def test_parse_fight_card() -> None:
    bouts = parse_fight_card(CARD)
    got = [
        (b.card_segment, b.weight_class_raw, b.fighter_a_name, b.fighter_b_name, b.source_section)
        for b in bouts
    ]
    assert got == [
        ("Main card", "Heavyweight", "Ciryl Gane", "Josh Hokit", "fight_card"),
        ("Main card", "Women's Bantamweight", "Kayla Harrison", "Amanda Nunes", "fight_card"),
        ("Preliminary card", "Welterweight", "Uroš Medić", "Kevin Holland", "fight_card"),
        (None, "Bantamweight", "Juan Díaz", "Alateng Heili", "announced_bouts"),
    ]  # duplicate announced bout skipped; "See also" list ignored
    assert [b.bout_order for b in bouts] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("dataset_latest", "today", "status"),
    [
        (date(2026, 10, 31), date(2026, 11, 7), "fresh"),
        (date(2026, 10, 24), date(2026, 11, 2), "pending"),  # 2 days after event, within grace
    ],
)
def test_staleness_ok(dataset_latest, today, status) -> None:
    past = [date(2026, 10, 31), date(2026, 10, 24)]
    assert check_staleness(dataset_latest, past, today, grace_days=4).status == status


def test_staleness_fails_loudly() -> None:
    with pytest.raises(StaleDataError, match="2026-10-31"):
        check_staleness(date(2026, 10, 24), [date(2026, 10, 31)], date(2026, 11, 7), grace_days=4)


def test_staleness_ignores_event_happening_today() -> None:
    r = check_staleness(
        date(2026, 10, 31), [date(2026, 11, 7), date(2026, 10, 31)], date(2026, 11, 7), grace_days=4
    )
    assert r.status == "fresh"


def test_staleness_without_reference_fails() -> None:
    with pytest.raises(StaleDataError):
        check_staleness(date(2026, 10, 31), [], date(2026, 11, 7), grace_days=4)
