"""The prediction log must be append-only, tamper-evident and pre-event only."""

import json
from datetime import UTC, datetime

import pytest

from ufc_model.tracking.predictions_log import GENESIS, ChainError, append, read, verify

NOW = datetime(2026, 11, 7, 19, 0, tzinfo=UTC)


def rec(i: int, start: str = "2026-11-07T22:00:00+00:00") -> dict:
    return {
        "event": "Test Night",
        "event_date": "2026-11-07",
        "event_start_utc": start,
        "bout_key": f"Test#{i}",
        "p_a": 0.6,
        "p_b": 0.4,
        "model_version": "lr-x",
    }


def test_chain_appends_and_verifies(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    append(path, [rec(1), rec(2)], now=NOW)
    append(path, [rec(3)], now=NOW)
    entries = read(path)
    assert [e["seq"] for e in entries] == [0, 1, 2]
    assert entries[0]["prev_hash"] == GENESIS
    assert entries[2]["prev_hash"] == entries[1]["hash"]
    assert verify(path) == 3


@pytest.mark.parametrize("mutation", ["edit", "delete", "reorder"])
def test_tampering_is_detected(tmp_path, mutation) -> None:
    path = tmp_path / "history.jsonl"
    append(path, [rec(1), rec(2), rec(3)], now=NOW)
    lines = path.read_text().splitlines()
    if mutation == "edit":
        e = json.loads(lines[1])
        e["p_a"] = 0.9
        lines[1] = json.dumps(e, sort_keys=True)
    elif mutation == "delete":
        del lines[1]
    else:
        lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ChainError):
        verify(path)
    with pytest.raises(ChainError):
        append(path, [rec(4)], now=NOW)


def test_refuses_logging_at_or_after_event_start(tmp_path) -> None:
    path = tmp_path / "history.jsonl"
    late = datetime(2026, 11, 7, 22, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="before the event"):
        append(path, [rec(1)], now=late)
    with pytest.raises(ValueError):
        append(path, [rec(1, "2026-11-08T00:00:00+00:00"), rec(2)], now=late)
    assert not path.exists() or read(path) == []  # nothing partially written
