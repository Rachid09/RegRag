"""Hosted text generation through LiteLLM, cached on disk, with retries on transient errors.

Gemini 3.6+ models ignore sampling parameters (and LiteLLM rejects `temperature` for the
Gemini 3 models we use), so no temperature is sent. Reproducibility comes from the cache:
the same (model, messages) always returns the stored answer, so reruns cost nothing.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from regrag import config
from regrag.cache import DiskCache, _log_backoff, cache_key

Messages = list[dict[str, str]]


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up; retrying today is pointless."""


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    seconds: float  # provider latency of the original call (kept in the cache for reports)


def is_transient(exc: BaseException) -> bool:
    """Per-minute 429s and provider overloads (5xx) are worth retrying; daily quotas are not."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "per day" not in str(exc).lower()
    return status in (500, 502, 503, 504)


def _litellm_complete(model: str, messages: Messages, max_tokens: int) -> tuple[str, int, int]:
    import litellm  # lazy: slow import, and tests inject a fake

    resp = litellm.completion(model=model, messages=messages, max_tokens=max_tokens)
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        (usage.prompt_tokens or 0) if usage else 0,
        (usage.completion_tokens or 0) if usage else 0,
    )


@dataclass
class LLM:
    model: str = config.GEN_MODELS[config.GEN_MODEL]
    max_tokens: int = 1024
    cache: DiskCache = field(default_factory=DiskCache)
    backend: Callable[[str, Messages, int], tuple[str, int, int]] = _litellm_complete

    def complete(self, messages: Messages) -> Completion:
        key = cache_key("chat", {"model": self.model, "messages": messages})
        hit = self.cache.get(key)
        if hit is not None:
            return Completion(
                hit["text"], self.model, hit["pt"], hit["ct"], True, hit.get("s", 0.0)
            )

        @retry(
            retry=retry_if_exception(is_transient),
            wait=wait_random_exponential(multiplier=2, max=60),
            stop=stop_after_attempt(6),
            reraise=True,
            before_sleep=_log_backoff,
        )
        def call():
            return self.backend(self.model, messages, self.max_tokens)

        start = time.monotonic()
        try:
            text, pt, ct = call()
        except Exception as exc:
            if getattr(exc, "status_code", None) == 429:
                raise QuotaExhausted(str(exc)[:300]) from exc
            raise
        seconds = time.monotonic() - start
        self.cache.set(key, {"text": text, "pt": pt, "ct": ct, "s": round(seconds, 3)})
        return Completion(text, self.model, pt, ct, False, seconds)


def make_llm(name: str | None = None) -> LLM:
    """`cheap` or `strong` (config.GEN_MODELS); defaults to config.GEN_MODEL."""
    return LLM(model=config.GEN_MODELS[name or config.GEN_MODEL])
