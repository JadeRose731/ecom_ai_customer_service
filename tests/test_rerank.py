import pytest

from app.core import rerank as rr


@pytest.mark.asyncio
async def test_rerank_orders_and_truncates(monkeypatch):
    async def fake_post(url, json, headers, timeout):
        class R:
            def raise_for_status(self): pass
            def json(self):
                return {"results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ]}
        return R()
    monkeypatch.setattr("app.core.rerank._post", fake_post)
    out = await rr.rerank("q", ["a", "b", "c"], top_n=2)
    assert out == [(1, 0.9), (2, 0.5)]


@pytest.mark.asyncio
async def test_rerank_empty_docs():
    assert await rr.rerank("q", []) == []
