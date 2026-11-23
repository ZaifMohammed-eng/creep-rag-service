"""LLM providers behind one interface, with retries, token and cost accounting.

* `AnthropicProvider`  - Anthropic Messages API (needs ANTHROPIC_API_KEY)
* `OpenAIProvider`     - OpenAI Chat Completions API (needs OPENAI_API_KEY)
* `ExtractiveProvider` - offline baseline: returns the context sentence that best
  matches the question. Used in tests, CI and when no key is configured.

Prices are not hard-coded because they change: set RAG_PRICE_IN_PER_MTOK and
RAG_PRICE_OUT_PER_MTOK (USD per million tokens) to get a cost per request.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from .text import split_sentences, tokenize

RETRYABLE = {408, 409, 429, 500, 502, 503, 504, 529}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def cost_usd(self) -> float | None:
        pin, pout = os.getenv("RAG_PRICE_IN_PER_MTOK"), os.getenv("RAG_PRICE_OUT_PER_MTOK")
        if pin is None or pout is None:
            return None
        return (self.input_tokens * float(pin) + self.output_tokens * float(pout)) / 1e6


class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse: ...


def _post_with_retries(client: httpx.Client, url: str, payload: dict, retries: int = 4,
                       backoff: float = 0.5) -> dict:
    for attempt in range(retries + 1):
        r = client.post(url, json=payload)
        if r.status_code < 400:
            return r.json()
        if r.status_code not in RETRYABLE or attempt == retries:
            r.raise_for_status()
        wait = float(r.headers.get("retry-after", backoff * 2 ** attempt))
        time.sleep(min(wait, 20.0))
    raise RuntimeError("unreachable")


class AnthropicProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, transport: httpx.BaseTransport | None = None):
        self.model = model or os.getenv("RAG_ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.name = f"anthropic:{self.model}"
        self.client = httpx.Client(
            base_url=base_url or os.getenv("RAG_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            headers={"x-api-key": api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
            timeout=60, transport=transport)

    def generate(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t = time.perf_counter()
        body = _post_with_retries(self.client, "/v1/messages", {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}]})
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        u = body.get("usage", {})
        return LLMResponse(text.strip(), self.model, u.get("input_tokens", 0), u.get("output_tokens", 0),
                           (time.perf_counter() - t) * 1000)


class OpenAIProvider:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None):
        self.model = model or os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"
        self.client = httpx.Client(
            base_url=os.getenv("RAG_OPENAI_BASE_URL", "https://api.openai.com"),
            headers={"Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY', '')}"},
            timeout=60, transport=transport)

    def generate(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t = time.perf_counter()
        body = _post_with_retries(self.client, "/v1/chat/completions", {
            "model": self.model, "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        u = body.get("usage", {})
        return LLMResponse(body["choices"][0]["message"]["content"].strip(), self.model,
                           u.get("prompt_tokens", 0), u.get("completion_tokens", 0),
                           (time.perf_counter() - t) * 1000)


_CTX = re.compile(r"\[(?P<id>[^\]]+)\]\s*(?P<text>.*?)(?=\n\[[^\]]+\]|\Z)", re.S)
_Q = re.compile(r"Question:\s*(?P<q>.*)", re.S)


class ExtractiveProvider:
    """No-LLM baseline that reads the same prompt the LLMs get.

    Scores every context sentence by IDF-free term overlap with the question and
    returns the best one with its citation. It cannot paraphrase or compute, so it is a
    floor for answer quality, not a competitor.
    """

    name = "extractive-baseline"
    model = "extractive"

    def generate(self, system: str, user: str, max_tokens: int = 300) -> LLMResponse:
        t = time.perf_counter()
        ctx, _, qpart = user.partition("Question:")
        q = set(tokenize(qpart))
        best, best_id, best_s, best_ov = "", "", -1e9, 0
        for rank, m in enumerate(_CTX.finditer(ctx.split("Context:", 1)[-1])):
            for s in split_sentences(m.group("text")):
                toks = tokenize(s)
                overlap = len(q & set(toks))
                # most shared terms wins; ties go to the higher-ranked passage, then the shorter sentence
                score = overlap - 0.05 * rank - 0.001 * len(toks)
                if score > best_s:
                    best, best_id, best_s, best_ov = s, m.group("id"), score, overlap
        text = f"{best} [{best_id}]" if best_ov > 0 else "INSUFFICIENT_CONTEXT"
        return LLMResponse(text, self.model, len(user.split()), len(text.split()),
                           (time.perf_counter() - t) * 1000)


def get_provider(kind: str | None = None) -> LLMProvider:
    kind = kind or os.getenv("RAG_LLM", "extractive")
    if kind == "anthropic":
        return AnthropicProvider()
    if kind == "openai":
        return OpenAIProvider()
    if kind == "extractive":
        return ExtractiveProvider()
    raise ValueError(f"unknown provider {kind!r}")
