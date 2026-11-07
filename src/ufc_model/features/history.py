"""Per-fighter fight history and point-in-time performance rates.

Every rate is pulled toward the league average *as of the fight date*
(``GlobalPriors``, updated only after each date is processed), so a fighter with
one short fight gets a sensible estimate instead of an extreme or missing value.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import date


@dataclass
class Totals:
    """Summed stats over a set of fights (own and opponent sides)."""

    minutes: float = 0.0
    sig_l: float = 0.0
    sig_a: float = 0.0
    opp_sig_l: float = 0.0
    opp_sig_a: float = 0.0
    td_l: float = 0.0
    td_a: float = 0.0
    opp_td_l: float = 0.0
    opp_td_a: float = 0.0
    sub_att: float = 0.0
    ctrl_s: float = 0.0
    kd: float = 0.0
    opp_kd: float = 0.0

    def add(self, other: Totals) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclass(frozen=True)
class FightRecord:
    """What a fighter's history keeps from one completed fight."""

    event_date: date
    outcome: str  # "W", "L", "D", "NC"
    method: str  # "ko", "sub", "dec", "other"
    totals: Totals | None  # None when the fight has no usable round stats

    @property
    def is_finish_win(self) -> bool:
        return self.outcome == "W" and self.method in {"ko", "sub"}

    @property
    def was_finished(self) -> bool:
        return self.outcome == "L" and self.method in {"ko", "sub"}


def method_category(method: str | None) -> str:
    m = (method or "").lower()
    if m.startswith("ko") or "tko" in m:
        return "ko"
    if m.startswith("submission"):
        return "sub"
    if m.startswith("decision"):
        return "dec"
    return "other"


@dataclass
class ShrinkConfig:
    minutes: float
    sig_attempts: float
    td_attempts: float


def _shrunk(num: float, den: float, prior: float, k: float) -> float:
    return (num + k * prior) / (den + k)


class GlobalPriors:
    """League-wide totals over all fights processed so far (point-in-time)."""

    def __init__(self) -> None:
        self.totals = Totals()

    def observe(self, t: Totals) -> None:
        self.totals.add(t)

    def rate(self, num: str, den: str, scale: float = 1.0, fallback: float = 0.0) -> float:
        d = getattr(self.totals, den)
        return scale * getattr(self.totals, num) / d if d > 0 else fallback


# Rough modern-UFC values used only before the first fight with stats exists.
_FALLBACK = {
    "slpm": 3.5,
    "acc": 0.45,
    "td15": 1.3,
    "tdacc": 0.38,
    "sub15": 0.5,
    "ctrl": 0.15,
    "kd15": 0.25,
}


def rate_features(
    t: Totals, priors: GlobalPriors, k: ShrinkConfig, prefix: str
) -> dict[str, float]:
    """Shrunk per-minute / accuracy / defense rates for one set of fights."""
    p = priors
    slpm = p.rate("sig_l", "minutes", fallback=_FALLBACK["slpm"])
    acc = p.rate("sig_l", "sig_a", fallback=_FALLBACK["acc"])
    td15 = p.rate("td_l", "minutes", 15, _FALLBACK["td15"])
    tdacc = p.rate("td_l", "td_a", fallback=_FALLBACK["tdacc"])
    sub15 = p.rate("sub_att", "minutes", 15, _FALLBACK["sub15"])
    ctrl = p.rate("ctrl_s", "minutes", 1 / 60, _FALLBACK["ctrl"])
    kd15 = p.rate("kd", "minutes", 15, _FALLBACK["kd15"])
    m = t.minutes
    return {
        f"{prefix}slpm": _shrunk(t.sig_l, m, slpm, k.minutes),
        f"{prefix}sapm": _shrunk(t.opp_sig_l, m, slpm, k.minutes),
        f"{prefix}sig_acc": _shrunk(t.sig_l, t.sig_a, acc, k.sig_attempts),
        f"{prefix}sig_def": 1 - _shrunk(t.opp_sig_l, t.opp_sig_a, acc, k.sig_attempts),
        f"{prefix}td_per15": _shrunk(15 * t.td_l, m, td15, k.minutes),
        f"{prefix}td_acc": _shrunk(t.td_l, t.td_a, tdacc, k.td_attempts),
        f"{prefix}td_def": 1 - _shrunk(t.opp_td_l, t.opp_td_a, tdacc, k.td_attempts),
        f"{prefix}sub_per15": _shrunk(15 * t.sub_att, m, sub15, k.minutes),
        f"{prefix}ctrl_share": _shrunk(t.ctrl_s / 60, m, ctrl, k.minutes),
        f"{prefix}kd_per15": _shrunk(15 * t.kd, m, kd15, k.minutes),
        f"{prefix}kd_absorbed_per15": _shrunk(15 * t.opp_kd, m, kd15, k.minutes),
    }


@dataclass
class FighterHistory:
    """Everything known about one fighter from fights before the current date."""

    records: list[FightRecord] = field(default_factory=list)
    career: Totals = field(default_factory=Totals)

    def add(self, rec: FightRecord) -> None:
        self.records.append(rec)
        if rec.totals is not None:
            self.career.add(rec.totals)

    def features(
        self, on: date, priors: GlobalPriors, k: ShrinkConfig, recent_n: int
    ) -> dict[str, float]:
        recs = self.records
        wins = sum(r.outcome == "W" for r in recs)
        losses = sum(r.outcome == "L" for r in recs)
        streak = 0
        for r in reversed(recs):
            if r.outcome == "NC":
                continue
            if (
                r.outcome == "D"
                or (streak > 0 and r.outcome == "L")
                or (streak < 0 and r.outcome == "W")
            ):
                break
            streak += 1 if r.outcome == "W" else -1
        decided_recent = [r for r in recs if r.outcome in {"W", "L"}][-recent_n:]
        recent_wins = sum(r.outcome == "W" for r in decided_recent)
        recent = Totals()
        for r in [r for r in recs if r.totals is not None][-recent_n:]:
            recent.add(r.totals)
        days = (on - recs[-1].event_date).days if recs else 0
        out = {
            "n_fights": float(len(recs)),
            "is_debut": float(not recs),
            "n_wins": float(wins),
            "n_losses": float(losses),
            "win_rate": (wins + 1) / (wins + losses + 2),
            "streak": float(streak),
            "finish_win_share": (sum(r.is_finish_win for r in recs) + 0.5) / (wins + 1),
            "been_finished_rate": (sum(r.was_finished for r in recs) + 0.25) / (len(recs) + 1),
            "recent_win_rate": (recent_wins + 1) / (len(decided_recent) + 2),
            "layoff_days": float(days),
            "cage_minutes": self.career.minutes,
        }
        out.update(rate_features(self.career, priors, k, ""))
        out.update(rate_features(recent, priors, k, "recent_"))
        return out
