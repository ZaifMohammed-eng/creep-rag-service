import json

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.llm import AnthropicProvider, ExtractiveProvider, OpenAIProvider
from rag.pipeline import SYSTEM, RAGPipeline, build_prompt


def anthropic_reply(text, status=200):
    return httpx.Response(status, json={"content": [{"type": "text", "text": text}],
                                        "usage": {"input_tokens": 120, "output_tokens": 8}})


def test_anthropic_payload_usage_and_cost(monkeypatch):
    seen = {}

    def handler(req: httpx.Request):
        seen["headers"], seen["body"] = req.headers, json.loads(req.content)
        return anthropic_reply("1912 [d1]")

    monkeypatch.setenv("RAG_PRICE_IN_PER_MTOK", "1.0")
    monkeypatch.setenv("RAG_PRICE_OUT_PER_MTOK", "5.0")
    p = AnthropicProvider(model="test-model", api_key="k", transport=httpx.MockTransport(handler))
    r = p.generate(SYSTEM, "Context:\n[d1] x\n\nQuestion: q")
    assert r.text == "1912 [d1]" and r.input_tokens == 120 and r.output_tokens == 8
    assert r.cost_usd == pytest.approx((120 * 1 + 8 * 5) / 1e6)
    assert seen["headers"]["x-api-key"] == "k" and seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == "test-model" and seen["body"]["temperature"] == 0
    assert seen["body"]["system"] == SYSTEM


def test_retries_on_overload_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return (httpx.Response(529, headers={"retry-after": "0"}) if calls["n"] < 3
                else anthropic_reply("ok [d1]"))

    p = AnthropicProvider(api_key="k", transport=httpx.MockTransport(handler))
    assert p.generate("s", "u").text == "ok [d1]" and calls["n"] == 3


def test_no_retry_on_client_error():
    p = OpenAIProvider(api_key="k", transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(httpx.HTTPStatusError):
        p.generate("s", "u")


def test_extractive_cites_best_sentence():
    prompt = build_prompt("When was the alloy invented?",
                          [("d1", "The alloy was invented in 1962. It is strong."), ("d2", "Cats sleep a lot.")])
    out = ExtractiveProvider().generate(SYSTEM, prompt).text
    assert out == "The alloy was invented in 1962. [d1]"


class FixedProvider:
    name, model = "fixed", "fixed"

    def __init__(self, text):
        self.text = text

    def generate(self, system, user, max_tokens=300):
        from rag.llm import LLMResponse
        return LLMResponse(self.text, "fixed", 10, 2)


def test_pipeline_rejects_uncited_and_fabricated_citations(retriever):
    for text, reason in [("Inconel 718", "ungrounded_answer"),
                         ("Inconel 718 [not_retrieved]", "ungrounded_answer"),
                         ("INSUFFICIENT_CONTEXT", "model_declined"),
                         ("Inconel 718 [inco]", "ok")]:
        a = RAGPipeline(retriever, FixedProvider(text), k=3).answer("nickel superalloy for jet engines")
        assert a.reason == reason
        assert (a.answer is not None) == (reason == "ok")


@pytest.fixture()
def client(retriever, tmp_path):
    app = create_app(RAGPipeline(retriever, ExtractiveProvider(), k=3), log_path=tmp_path / "log.jsonl")
    return TestClient(app), tmp_path / "log.jsonl"


def test_query_endpoint_cache_log_metrics(client):
    c, log = client
    body = {"query": "What does BM25 score documents by?"}
    r1, r2 = c.post("/v1/query", json=body).json(), c.post("/v1/query", json=body).json()
    assert r1["citations"] == ["bm25"] and not r1["abstained"]
    assert r1["cached"] is False and r2["cached"] is True
    assert len(log.read_text().splitlines()) == 1
    m = c.get("/metrics").text
    assert 'rag_requests_total{outcome="ok"} 1.0' in m and "rag_cache_hits_total 1.0" in m


def test_validation_and_ingest(client):
    c, _ = client
    assert c.post("/v1/query", json={"query": "x"}).status_code == 422
    r = c.post("/v1/documents", json={"documents": [
        {"doc_id": "ti64", "title": "Ti-6Al-4V", "text": "Ti-6Al-4V is an alpha-beta titanium alloy used in aerospace fasteners."}]})
    assert r.json()["added_chunks"] == 1
    assert c.post("/v1/documents", json={"documents": [{"doc_id": "ti64", "text": "dup"}]}).status_code == 409
    a = c.post("/v1/query", json={"query": "Which titanium alloy is alpha-beta?"}).json()
    assert a["citations"] == ["ti64#0"]
    assert c.get("/healthz").json()["documents"] == 6
