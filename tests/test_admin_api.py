# ch03 Task 17:后台聚合首页——依赖全挂也得出全部卡片,各自报自己读不到
from app.db import repository


async def test_overview_returns_all_cards_even_when_deps_down(client, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("依赖挂了")

    for fn in ("knowledge_stats", "staging_stats", "list_conversations_with_messages"):
        monkeypatch.setattr(f"app.api.admin.{fn}", boom)
    monkeypatch.setattr("app.api.observability.list_eval_runs", boom)   # ch09:趋势块同款
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert len(cards) == 6  # ch04 起 + RAG 评估卡 + ch09 观测卡
    by_key = {c["key"]: c for c in cards}
    # 依赖库的三张卡连坐报读不到;RAG 评估/观测卡只读文件,自己报自己的态
    for key in ("chat", "kb", "staging"):
        assert by_key[key]["state"] == "unreachable"
    assert by_key["rageval"]["state"] in ("empty", "ok")
    assert by_key["observability"]["state"] == "unreachable"   # list_eval_runs 挂 → 观测卡连坐


async def test_overview_reports_kb_numbers(client, db_session_factory, db_clean):
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = {c["key"]: c for c in r.json()["cards"]}
    assert cards["kb"]["state"] == "empty"  # 没数据是一种状态,不是错误
