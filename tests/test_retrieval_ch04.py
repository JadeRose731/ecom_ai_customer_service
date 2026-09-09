# ch04:retrieval 四策略分支 + 首尾组装(子步全 mock,不打真实上游)
import pytest

from app.core import retrieval


def test_arrange_head_tail():
    # 输入按相关度降序 [a,b,c,d,e] → a 首、b 尾、剩 c,d,e 居中
    out = retrieval.arrange_head_tail(["a", "b", "c", "d", "e"])
    assert out[0] == "a" and out[-1] == "b"
    assert set(out[1:-1]) == {"c", "d", "e"}


def test_arrange_head_tail_short_input_kept():
    assert retrieval.arrange_head_tail([]) == []
    assert retrieval.arrange_head_tail(["a"]) == ["a"]
    assert retrieval.arrange_head_tail(["a", "b"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_strategy_vector(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    called = {}
    def fake_dense(client, vec, top_k, category=None, collection="knowledge"):
        called["dense"] = True
        return [{"id": 1, "score": 0.9, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.kb.milvus_client.dense_search", fake_dense)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("运费", strategy="vector", client=object())
    assert called.get("dense") and out[0]["id"] == 1


@pytest.mark.asyncio
async def test_strategy_bm25_passes_query_text(monkeypatch):
    seen = {}
    def fake_bm25(client, text, top_k, category=None, collection="knowledge"):
        seen["text"] = text
        return []
    monkeypatch.setattr("app.kb.milvus_client.bm25_search", fake_bm25)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    out = await retrieval.search_knowledge("猫砂盆 Pro", strategy="bm25", client=object())
    assert seen["text"] == "猫砂盆 Pro" and out == []


@pytest.mark.asyncio
async def test_strategy_hybrid_rerank(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [
            {"id": 1, "score": 0.5, "question": "运费", "answer": "满99包邮",
             "section_path": "p1", "content_type": "faq", "category": "运费"},
            {"id": 2, "score": 0.4, "question": "型号", "answer": "Pro自动清理",
             "section_path": "p2", "content_type": "manual", "category": "商品"},
        ]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    async def fake_rerank(q, docs, top_n=None):
        return [(1, 0.95), (0, 0.2)]  # 第 2 条(index1)最相关
    monkeypatch.setattr("app.core.rerank.rerank", fake_rerank)
    out = await retrieval.search_knowledge("Pro 型号", strategy="hybrid_rerank", client=object())
    assert out[0]["id"] == 2 and out[0]["rerank_score"] == 0.95


@pytest.mark.asyncio
async def test_strategy_hybrid_truncates_without_rerank(monkeypatch):
    async def fake_embed(q): return [0.1] * 1024
    monkeypatch.setattr("app.core.embeddings.embed_query", fake_embed)
    monkeypatch.setattr("app.kb.milvus_client.ensure_collection", lambda *a, **k: None)
    def fake_hybrid(client, vec, text, top_k, recall=50, category=None, collection="knowledge"):
        return [{"id": i, "score": 0.5, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "c"}
                for i in range(20)]
    monkeypatch.setattr("app.kb.milvus_client.hybrid_search", fake_hybrid)
    called = {}
    async def fail_rerank(q, docs, top_n=None):
        called["rerank"] = True
        return []
    monkeypatch.setattr("app.core.rerank.rerank", fail_rerank)
    out = await retrieval.search_knowledge("q", strategy="hybrid", client=object())
    assert "rerank" not in called and len(out) <= 10  # hybrid 不重排,只截 top_k
