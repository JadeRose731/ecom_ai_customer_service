# tests/test_tool_faq.py
# ch03 起 query_faq 内部走向量检索(契约不变);旧查表断言按计划改造为对
# retrieval.search_knowledge 的 mock 断言,不再依赖 faq 表内容。
from app.tools.business import query_faq

async def test_query_faq_hit(monkeypatch):
    async def fake_search(keyword, **kw):
        assert keyword == "退货"
        return [{"id": 1, "score": 0.8, "question": "退货政策", "answer": "7 天无理由"}]
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    r = await query_faq.ainvoke({"keyword": "退货"})
    assert r["hits"] and r["hits"][0]["question"] == "退货政策"

async def test_query_faq_miss_returns_message(monkeypatch):
    async def fake_search(keyword, **kw):
        return []
    monkeypatch.setattr("app.tools.business.retrieval.search_knowledge", fake_search)
    r = await query_faq.ainvoke({"keyword": "鞋子"})
    assert r["hits"] == [] and "未找到" in r["message"]
