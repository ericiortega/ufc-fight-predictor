"""The HTTP client must fail loudly, never return challenge pages or partial data."""

import pytest
import requests

from ufc_model.config import HttpConfig
from ufc_model.http import BlockedError, FetchError, PoliteClient

CFG = HttpConfig(min_delay_s=0, max_delay_s=0, timeout_s=1, max_retries=2)


class _Resp:
    def __init__(self, status: int, text: str = "ok") -> None:
        self.status_code, self.text = status, text


@pytest.fixture
def client(monkeypatch) -> PoliteClient:
    monkeypatch.setattr("ufc_model.http.time.sleep", lambda s: None)
    return PoliteClient(CFG, "test-agent", seed=0)


def test_user_agent_has_no_personal_data() -> None:
    from ufc_model.config import get_config

    ua = get_config().project.user_agent
    assert "@" not in ua and "github.com" in ua


def test_bot_challenge_raises(client, monkeypatch) -> None:
    page = "<p>Checking your browser…</p><noscript>This site requires JavaScript.</noscript>"
    monkeypatch.setattr(client.session, "get", lambda *a, **k: _Resp(200, page))
    with pytest.raises(BlockedError):
        client.get("http://example.com")


def test_http_error_raises_after_retries(client, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(client.session, "get", lambda *a, **k: calls.append(1) or _Resp(503))
    with pytest.raises(FetchError, match="503"):
        client.get("http://example.com")
    assert len(calls) == 2


def test_connection_error_raises(client, monkeypatch) -> None:
    def boom(*a, **k):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(client.session, "get", boom)
    with pytest.raises(FetchError, match="refused"):
        client.get("http://example.com")


def test_404_is_not_retried(client, monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(client.session, "get", lambda *a, **k: calls.append(1) or _Resp(404))
    with pytest.raises(FetchError, match="404"):
        client.get("http://example.com")
    assert len(calls) == 1
