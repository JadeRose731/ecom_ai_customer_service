# ch04:query_faq 升级为 RAG 管线(理解→混合重排→两道证据闸→首尾组装编号证据)。
# 子步全 mock;管线里任一真实上游(嵌入/重排/聊天)都没 key,真实链路见 make eval-rag。
import pytest

from app.tools import business


@pytest.mark.asyncio
async def test_query_faq_sufficient(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": ["运费"]}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, strategy="hybrid_rerank", top_k=None, category=None, client=None, collection="knowledge"):
        return [
            {"id": 1, "rerank_score": 0.9, "question": "运费", "answer": "满99包邮",
             "section_path": "运费政策", "content_type": "faq", "category": "运费"},
            {"id": 2, "rerank_score": 0.6, "question": "退货", "answer": "七天无理由",
             "section_path": "退货政策", "content_type": "policy", "category": "退货"},
        ]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": True, "reason": "够"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)

    out = await business.query_faq.ainvoke({"keyword": "邮费多少"})
    assert out["sufficient"] is True
    assert out["citations"][0]["n"] == 1 and out["citations"][0]["section_path"]
    assert "[1]" in out["evidence"]


@pytest.mark.asyncio
async def test_query_faq_retrieval_low_conf(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, **kw): return []  # 无召回
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "火星车怎么买"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"


@pytest.mark.asyncio
async def test_query_faq_self_check_fail(monkeypatch):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, **kw):
        return [{"id": 1, "rerank_score": 0.8, "question": "运费", "answer": "满99包邮",
                 "section_path": "p", "content_type": "faq", "category": "运费"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": False, "reason": "问型号但证据只讲运费"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)
    out = await business.query_faq.ainvoke({"keyword": "Pro型号能自动铲屎吗"})
    assert out["sufficient"] is False and out["source"] == "self_check"
    assert "型号" in out["reason"]


@pytest.mark.asyncio
async def test_query_faq_low_rerank_score_refuses(monkeypatch):
    """机械闸:召回了但最高重排分低于阈值,同样判证据不足(retrieval_low_conf)。"""
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    async def fake_search(query, **kw):
        return [{"id": 1, "rerank_score": 0.05, "question": "q", "answer": "a",
                 "section_path": "p", "content_type": "faq", "category": "c"}]
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    out = await business.query_faq.ainvoke({"keyword": "随便什么"})
    assert out["sufficient"] is False and out["source"] == "retrieval_low_conf"
