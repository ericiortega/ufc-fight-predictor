"""Fail loudly when the upstream dataset lags behind real-world UFC events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

log = logging.getLogger(__name__)


class StaleDataError(RuntimeError):
    """Upstream data has not been refreshed since the last completed UFC event."""


@dataclass(frozen=True)
class StalenessReport:
    dataset_latest: date
    last_completed_event: date | None
    days_since_event: int | None
    status: str  # "fresh", "pending" (within grace), or "stale"


def check_staleness(
    dataset_latest: date,
    past_event_dates: list[date],
    today: date,
    grace_days: int,
) -> StalenessReport:
    """Compare the dataset's newest event with the last event Wikipedia lists as past.

    Events dated today are ignored (they may still be in progress). Raises
    :class:`StaleDataError` once the gap is older than ``grace_days``.
    """
    completed = [d for d in past_event_dates if d < today]
    if not completed:
        raise StaleDataError(
            "No completed events found in the reference list; cannot verify freshness"
        )
    last = max(completed)
    gap = (today - last).days
    if dataset_latest >= last:
        return StalenessReport(dataset_latest, last, gap, "fresh")
    if gap <= grace_days:
        log.warning(
            "Upstream data ends %s but an event happened %s; within %d-day grace period",
            dataset_latest,
            last,
            grace_days,
        )
        return StalenessReport(dataset_latest, last, gap, "pending")
    raise StaleDataError(
        f"Upstream dataset ends {dataset_latest}, but a UFC event was completed on {last} "
        f"({gap} days ago, grace {grace_days}). Upstream may have stopped updating."
    )
