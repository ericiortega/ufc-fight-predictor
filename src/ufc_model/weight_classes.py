"""Canonical weight classes shared by the ufcstats and Wikipedia parsers."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordered so that longer names match first ("Light Heavyweight" before "Heavyweight").
_CLASSES: list[tuple[str, int | None]] = [
    ("Women's Strawweight", 115),
    ("Women's Flyweight", 125),
    ("Women's Bantamweight", 135),
    ("Women's Featherweight", 145),
    ("Light Heavyweight", 205),
    ("Heavyweight", 265),
    ("Strawweight", 115),
    ("Flyweight", 125),
    ("Bantamweight", 135),
    ("Featherweight", 145),
    ("Lightweight", 155),
    ("Welterweight", 170),
    ("Middleweight", 185),
    ("Catch Weight", None),
    ("Open Weight", None),
]

WEIGHT_CLASS_LBS: dict[str, int | None] = dict(_CLASSES)


@dataclass(frozen=True)
class WeightClass:
    name: str | None
    limit_lbs: int | None
    is_women: bool
    is_title: bool


def parse_weight_class(raw: str | None) -> WeightClass:
    """Map strings like "UFC Interim Women's Flyweight Title Bout" to a canonical class."""
    text = raw.strip() if isinstance(raw, str) else ""
    norm = (
        re.sub(r"\s+", " ", text.replace("’", "'")).lower().replace("catchweight", "catch weight")
    )
    is_title = "title" in norm or "championship" in norm
    name: str | None = None
    for cls, _ in _CLASSES:
        if cls.lower() in norm:
            name = cls
            break
    if name is None and "superfight" in norm:
        name = "Open Weight"
    if name is not None and "women's" in norm and not name.startswith("Women's"):
        name = f"Women's {name}"
    limit = WEIGHT_CLASS_LBS.get(name) if name else None
    return WeightClass(
        name=name,
        limit_lbs=limit,
        is_women=bool(name and name.startswith("Women's")),
        is_title=is_title,
    )
