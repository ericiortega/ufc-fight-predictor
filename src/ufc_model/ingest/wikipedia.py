"""Upcoming UFC events and fight cards from Wikipedia (via the MediaWiki API).

Wikipedia text is CC BY-SA 4.0. Every fetched page is cached with its revision ID
and retrieval time, so each upcoming card is traceable to an exact revision.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag

from ufc_model.config import Config
from ufc_model.http import PoliteClient
from ufc_model.names import clean_display_name

log = logging.getLogger(__name__)


@dataclass
class WikiPage:
    title: str
    revid: int
    html: str
    retrieved_at: str


@dataclass
class ScheduledEvent:
    name: str
    event_date: date
    venue: str | None
    location: str | None
    wiki_title: str | None


@dataclass
class CardBout:
    bout_order: int
    card_segment: str | None
    weight_class_raw: str
    fighter_a_name: str
    fighter_b_name: str
    source_section: str  # "fight_card" or "announced_bouts"
    is_title: bool = False  # a fighter is marked "(c)" / "(ic)" (champion / interim)


def fetch_page(client: PoliteClient, cfg: Config, title: str, cache_dir: Path) -> WikiPage:
    """Fetch the parsed HTML of the current revision of ``title`` and cache it."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text|revid",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    }
    data = client.get(cfg.wikipedia.api_url, params=params).json()
    if "error" in data:
        raise ValueError(f"Wikipedia API error for {title!r}: {data['error'].get('info')}")
    parsed = data["parse"]
    page = WikiPage(
        title=parsed["title"],
        revid=parsed["revid"],
        html=parsed["text"],
        retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", page.title)
    (cache_dir / f"{safe}__rev{page.revid}.json").write_text(json.dumps(asdict(page)))
    return page


def _cell_text(cell: Tag) -> str:
    for sup in cell.find_all("sup"):
        sup.decompose()
    return re.sub(r"\s+", " ", cell.get_text(" ", strip=True)).replace(" ,", ",").strip()


def _parse_list_date(text: str) -> date | None:
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_events_list(html: str) -> tuple[list[ScheduledEvent], list[date]]:
    """Parse ``List_of_UFC_events`` into (scheduled events, past event dates)."""
    soup = BeautifulSoup(html, "lxml")
    scheduled: list[ScheduledEvent] = []
    table = soup.find("table", id="Scheduled_events")
    if table is None:
        raise ValueError("Scheduled_events table not found; page layout changed?")
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"])
        if len(cells) < 4:
            continue
        link = cells[0].find("a", href=True)
        title = None
        if link and link["href"].startswith("/wiki/"):
            title = unquote(link["href"][len("/wiki/") :]).replace("_", " ")
        when = _parse_list_date(_cell_text(cells[1]))
        if when is None:
            log.warning("Skipping scheduled row with unparseable date: %s", _cell_text(cells[1]))
            continue
        scheduled.append(
            ScheduledEvent(
                name=_cell_text(cells[0]),
                event_date=when,
                venue=_cell_text(cells[2]) or None,
                location=_cell_text(cells[3]) or None,
                wiki_title=title,
            )
        )
    past_dates: list[date] = []
    past = soup.find("table", id="Past_events")
    if past is None:
        raise ValueError("Past_events table not found; page layout changed?")
    for tr in past.find_all("tr")[1:]:
        cells = tr.find_all(["th", "td"])
        if len(cells) >= 3 and (d := _parse_list_date(_cell_text(cells[2]))):
            past_dates.append(d)
    return sorted(scheduled, key=lambda e: e.event_date), past_dates


_ANNOUNCED_RE = re.compile(
    r"^(?P<wc>.+?)\s+bout:\s*(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)\s*$", re.IGNORECASE
)


def parse_fight_card(html: str) -> list[CardBout]:
    """Extract bouts from fight-card tables and the "Announced bouts" list."""
    soup = BeautifulSoup(html, "lxml")
    bouts: list[CardBout] = []
    seen: set[frozenset[str]] = set()

    def add(segment: str | None, wc: str, a: str, b: str, section: str) -> None:
        is_title = (
            bool(re.search(r"\((?:c|ic)\)", f"{a} {b}", re.IGNORECASE)) or "title" in wc.lower()
        )
        a, b = clean_display_name(a), clean_display_name(b)
        key = frozenset((a.lower(), b.lower()))
        if not a or not b or key in seen:
            return
        seen.add(key)
        bouts.append(CardBout(len(bouts) + 1, segment, wc.strip(), a, b, section, is_title))

    for table in soup.find_all("table", class_="toccolours"):
        segment = None
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            texts = [_cell_text(c) for c in cells]
            if len(cells) == 1:
                segment = re.sub(r"\s*\(.*\)$", "", texts[0]) or None
            elif len(texts) >= 4 and texts[2].lower() in {"vs.", "vs"}:
                add(segment, texts[0], texts[1], texts[3], "fight_card")

    heading = soup.find(id="Announced_bouts")
    if heading is not None:
        container = heading.find_parent("div", class_="mw-heading") or heading
        sib = container.find_next_sibling()
        while (
            sib is not None
            and sib.name not in {"h2", "h3"}
            and "mw-heading" not in (sib.get("class") or [])
        ):
            if sib.name == "ul":
                for li in sib.find_all("li", recursive=False):
                    if m := _ANNOUNCED_RE.match(_cell_text(li)):
                        add(None, m["wc"], m["a"], m["b"], "announced_bouts")
            sib = sib.find_next_sibling()
    return bouts
