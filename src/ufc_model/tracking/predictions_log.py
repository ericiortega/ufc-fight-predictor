"""Append-only, hash-chained prediction log (``predictions/history.jsonl``).

Each line is a JSON object with ``seq``, ``prev_hash`` and ``hash``, where
``hash = sha256(canonical JSON of the line without "hash")`` and ``prev_hash`` is
the previous line's hash (64 zeros for the first). Editing, deleting or
reordering any past line breaks the chain, and git history dates every append.
Predictions are only logged for events that have not started.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

GENESIS = "0" * 64


class ChainError(RuntimeError):
    """The log has been modified: a hash or link no longer matches."""


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify(path: Path) -> int:
    """Raise :class:`ChainError` if any line was altered; return the number of entries."""
    prev = GENESIS
    entries = read(path)
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            raise ChainError(f"line {i}: seq {e.get('seq')} != {i}")
        if e.get("prev_hash") != prev:
            raise ChainError(f"line {i}: prev_hash does not match line {i - 1}")
        if e.get("hash") != _digest(e):
            raise ChainError(f"line {i}: content does not match its hash")
        prev = e["hash"]
    return len(entries)


def append(path: Path, records: list[dict], now: datetime | None = None) -> list[dict]:
    """Verify the existing chain, then append ``records``.

    Each record carries ``event_start_utc``; logging at or after it is refused.
    """
    now = now or datetime.now(UTC)
    for rec in records:  # check everything before writing anything
        if now >= datetime.fromisoformat(rec["event_start_utc"]):
            raise ValueError(
                f"{rec['event']} starts {rec['event_start_utc']}; "
                "predictions must be logged before the event"
            )
    n = verify(path)
    entries = read(path)
    prev = entries[-1]["hash"] if entries else GENESIS
    written = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            e = {"seq": n, "logged_at": now.isoformat(timespec="seconds"), **rec, "prev_hash": prev}
            e["hash"] = _digest(e)
            f.write(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n")
            prev, n = e["hash"], n + 1
            written.append(e)
    return written


def latest_pre_event(entries: list[dict]) -> dict[str, dict]:
    """The last prediction logged for each bout key (what gets scored)."""
    out: dict[str, dict] = {}
    for e in entries:
        out[e["bout_key"]] = e
    return out
