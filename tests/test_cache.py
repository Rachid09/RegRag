import pytest

from regrag.cache import DiskCache, cache_key, retry_on_429


class FakeHTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(status_code)
        self.status_code = status_code


def test_cache_key_is_order_independent_for_dicts():
    assert cache_key("ns", {"a": 1, "b": 2}) == cache_key("ns", {"b": 2, "a": 1})
    assert cache_key("ns", {"a": 1}) != cache_key("other", {"a": 1})


def test_cached_computes_once_and_survives_reopen(tmp_path):
    path = tmp_path / "c.sqlite"
    calls = []

    def compute():
        calls.append(1)
        return {"vector": [0.1, 0.2]}

    assert DiskCache(path).cached("embed", {"t": "x"}, compute) == {"vector": [0.1, 0.2]}
    assert DiskCache(path).cached("embed", {"t": "x"}, compute) == {"vector": [0.1, 0.2]}
    assert len(calls) == 1


def test_retry_on_429_then_succeeds():
    attempts = []

    @retry_on_429(attempts=3, max_wait=0)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise FakeHTTPError(429)
        return "ok"

    assert flaky() == "ok"
    assert len(attempts) == 3


def test_retry_does_not_retry_other_errors():
    attempts = []

    @retry_on_429(attempts=5, max_wait=0)
    def broken():
        attempts.append(1)
        raise FakeHTTPError(400)

    with pytest.raises(FakeHTTPError):
        broken()
    assert len(attempts) == 1


def test_retry_gives_up_after_attempts():
    @retry_on_429(attempts=2, max_wait=0)
    def always_429():
        raise FakeHTTPError(429)

    with pytest.raises(FakeHTTPError):
        always_429()


def test_daily_quota_429_is_not_retried():
    from regrag.cache import is_rate_limited

    minute = Exception("Quota exceeded: requests per minute")
    minute.status_code = 429
    daily = Exception("Embed content requests per day limit exceeded on Free Tier")
    daily.status_code = 429
    assert is_rate_limited(minute)
    assert not is_rate_limited(daily)
