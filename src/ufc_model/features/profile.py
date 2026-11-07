"""Point-in-time imputation of profile fields (height, reach, stance, age).

Profiles are scraped *today*, and missing values correlate with short careers
(often ended by losses), which is future information. So the model never sees
NaN or a missingness indicator for these fields. Gaps are filled only from
fighters observed in fights *before* the current date:

- reach:  height + mean(reach - height), if height is known; else class median reach
- height: weight-class median height
- stance: weight-class most common stance
- age:    weight-class median age at fight time

Imputed flags are returned for reporting only and never become features.
"""

from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date

ALL = "__all__"
STANCES = ("orthodox", "southpaw", "switch", "other")


def stance_category(raw: str | None) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    s = raw.strip().lower()
    return s if s in STANCES[:3] else "other"


@dataclass(frozen=True)
class Profile:
    height_in: float | None
    reach_in: float | None
    stance: str | None
    dob: date | None


@dataclass(frozen=True)
class ImputedProfile:
    height_in: float
    reach_in: float
    stance: str
    age_years: float
    imputed: dict[str, bool]  # reporting only; never a model feature


def _median(values: list[float]) -> float | None:
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2


def age_years(dob: date, on: date) -> float:
    return (on - dob).days / 365.25


class ProfileImputer:
    """Accumulates profile statistics from fighters already seen, by weight class."""

    def __init__(
        self, default_height: float, default_reach: float, default_age: float, default_stance: str
    ) -> None:
        self.defaults = {"height": default_height, "reach": default_reach, "age": default_age}
        self.default_stance = default_stance
        self._heights: dict[str, list[float]] = defaultdict(list)  # sorted
        self._reaches: dict[str, list[float]] = defaultdict(list)
        self._ages: dict[str, list[float]] = defaultdict(list)
        self._stances: dict[str, Counter[str]] = defaultdict(Counter)
        self._seen: set[tuple[str, str]] = set()
        self._ape_sum = 0.0
        self._ape_n = 0

    @staticmethod
    def _keys(weight_class: str | None) -> list[str]:
        return (
            [ALL] if weight_class in (None, "Catch Weight", "Open Weight") else [weight_class, ALL]
        )

    def _class_median(self, store: dict[str, list[float]], wc: str | None, kind: str) -> float:
        for key in self._keys(wc):
            if (m := _median(store[key])) is not None:
                return m
        return self.defaults[kind]

    def impute(self, p: Profile, weight_class: str | None, on: date) -> ImputedProfile:
        """Fill gaps using only statistics observed before ``on``."""
        flags = {
            "height": p.height_in is None,
            "reach": p.reach_in is None,
            "stance": p.stance is None,
            "age": p.dob is None,
        }
        height = (
            p.height_in
            if p.height_in is not None
            else self._class_median(self._heights, weight_class, "height")
        )
        if p.reach_in is not None:
            reach = p.reach_in
        elif p.height_in is not None:
            offset = self._ape_sum / self._ape_n if self._ape_n else 0.0
            reach = p.height_in + offset
        else:
            reach = self._class_median(self._reaches, weight_class, "reach")
        stance = p.stance
        if stance is None:
            stance = self.default_stance
            for key in self._keys(weight_class):
                if self._stances[key]:
                    stance = self._stances[key].most_common(1)[0][0]
                    break
        age = (
            age_years(p.dob, on)
            if p.dob is not None
            else self._class_median(self._ages, weight_class, "age")
        )
        # Real heights and reaches are whole inches; an imputed 71.47" would make
        # imputed rows (and so short careers) identifiable to the model.
        if flags["height"]:
            height = float(round(height))
        if flags["reach"]:
            reach = float(round(reach))
        return ImputedProfile(height, reach, stance, age, flags)

    def observe(
        self, fighter_id: str | None, p: Profile, weight_class: str | None, on: date
    ) -> None:
        """Record a fighter who competed on ``on``. Call only after that date is processed."""
        for key in self._keys(weight_class):
            if p.dob is not None:
                bisect.insort(self._ages[key], age_years(p.dob, on))
            if fighter_id is None or (fighter_id, key) in self._seen:
                continue
            self._seen.add((fighter_id, key))
            if p.height_in is not None:
                bisect.insort(self._heights[key], p.height_in)
            if p.reach_in is not None:
                bisect.insort(self._reaches[key], p.reach_in)
            if p.stance is not None:
                self._stances[key][p.stance] += 1
            if key == ALL and p.height_in is not None and p.reach_in is not None:
                self._ape_sum += p.reach_in - p.height_in
                self._ape_n += 1
