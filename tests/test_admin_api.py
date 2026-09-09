# ch03 Task 17:后台聚合首页——依赖全挂也得出全部卡片,各自报自己读不到
from app.db import repository


async def test_overview_returns_all_cards_even_when_deps_down(client, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("依赖挂了")

    for fn in ("knowledge_stats", "staging_stats", "list_conversations_with_messages"):
        monkeypatch.setattr(f"app.api.admin.{fn}", boom)
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert len(cards) == 4
    assert all(c["state"] == "unreachable" for c in cards)


async def test_overview_reports_kb_numbers(client, db_session_factory, db_clean):
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = {c["key"]: c for c in r.json()["cards"]}
    assert cards["kb"]["state"] == "empty"  # 没数据是一种状态,不是错误
