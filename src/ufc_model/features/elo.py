"""Elo ratings updated after each fight.

Draws score 0.5; no-contests do not change ratings. A fighter whose identity is
unresolved plays at the initial rating and is never stored, so their opponent's
rating still updates against a neutral, uninformative opponent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloParams:
    initial: float = 1500.0
    k: float = 32.0
    finish_multiplier: float = 1.0  # extra K for KO/TKO/submission results


def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B under the Elo model."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class EloRatings:
    """Mutable rating table keyed by fighter_id."""

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self.ratings: dict[str, float] = {}

    def get(self, fighter_id: str | None) -> float:
        if fighter_id is None:
            return self.params.initial
        return self.ratings.get(fighter_id, self.params.initial)

    def update(
        self, a: str | None, b: str | None, score_a: float, finish: bool
    ) -> tuple[float, float]:
        """Apply one result (``score_a`` in {0, 0.5, 1}); returns the new ratings."""
        ra, rb = self.get(a), self.get(b)
        k = self.params.k * (self.params.finish_multiplier if finish else 1.0)
        delta = k * (score_a - expected_score(ra, rb))
        ra, rb = ra + delta, rb - delta
        if a is not None:
            self.ratings[a] = ra
        if b is not None:
            self.ratings[b] = rb
        return ra, rb
