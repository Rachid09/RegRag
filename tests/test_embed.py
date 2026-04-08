import pytest

from regrag.cache import DiskCache
from regrag.embed import Embedder, decode_vector, encode_vector, query_text


class FakeBackend:
    def __init__(self, fail_first_with_429=0):
        self.calls: list[list[str]] = []
        self.fail = fail_first_with_429

    def __call__(self, texts, model, dims):
        if self.fail:
            self.fail -= 1
            err = Exception("rate limited")
            err.status_code = 429
            raise err
        self.calls.append(list(texts))
        return [[float(len(t)), float(dims)] for t in texts], 10 * len(texts)


def make(tmp_path, backend, batch=2):
    return Embedder(
        model="m",
        dims=3,
        batch_size=batch,
        rpm=0,
        cache=DiskCache(tmp_path / "c.sqlite"),
        backend=backend,
    )


def test_vector_roundtrip_float32():
    v = [0.125, -1.5, 3.0]
    assert decode_vector(encode_vector(v)) == v


def test_query_prefix():
    assert query_text("Who?") == "task: search result | query: Who?"


def test_batches_and_preserves_order(tmp_path):
    be = FakeBackend()
    e = make(tmp_path, be, batch=2)
    out = e.embed(["a", "bbb", "cc"])
    assert out == [[1.0, 3.0], [3.0, 3.0], [2.0, 3.0]]
    assert be.calls == [["a", "bbb"], ["cc"]]
    assert e.requests_made == 2 and e.tokens_used == 30


def test_cache_hits_skip_api_and_only_misses_are_sent(tmp_path):
    be = FakeBackend()
    make(tmp_path, be).embed(["a", "bbb"])
    be2 = FakeBackend()
    e2 = make(tmp_path, be2)
    assert e2.embed(["bbb", "new", "a"]) == [[3.0, 3.0], [3.0, 3.0], [1.0, 3.0]]
    assert be2.calls == [["new"]]


def test_cache_is_keyed_by_model_and_dims(tmp_path):
    be = FakeBackend()
    make(tmp_path, be).embed(["a"])
    other = Embedder(
        model="m", dims=7, batch_size=2, rpm=0, cache=DiskCache(tmp_path / "c.sqlite"), backend=be
    )
    other.embed(["a"])
    assert be.calls == [["a"], ["a"]]


def test_retries_429_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("tenacity.nap.time.sleep", lambda s: None)
    be = FakeBackend(fail_first_with_429=2)
    assert make(tmp_path, be).embed(["a"]) == [[1.0, 3.0]]


def test_wrong_vector_count_raises(tmp_path):
    def short(texts, model, dims):
        return [[0.0]], 1

    with pytest.raises(RuntimeError, match="asked for 2"):
        make(tmp_path, short).embed(["a", "b"])
