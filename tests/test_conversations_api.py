# tests/test_conversations_api.py — ch07 会话侧栏两个只读接口(路由层打桩,不触真库;
# 本仓惯例:API 层 mock、repository 层真库——TestClient 的 anyio portal 与 pytest-asyncio
# 事件循环不共库连接,API 测试直连真库会报 attached to a different loop)
from datetime import datetime

from app.api import conversations


async def test_list_conversations_passes_user_id_and_serializes(client, monkeypatch):
    seen = {}

    async def fake_list(user_id, limit=50):
        seen["user_id"] = user_id
        return [{"id": 7, "status": "进行中", "preview": "订单1001到哪了",
                 "has_summary": True, "updated_at": "2026-09-18T10:00:00"}]

    monkeypatch.setattr(conversations.repository, "list_conversations", fake_list)
    r = await client.get("/api/conversations", params={"user_id": "u-42"})
    assert r.status_code == 200
    assert seen["user_id"] == "u-42"                        # user_id 透传
    (item,) = r.json()["items"]
    assert item["id"] == 7 and item["status"] == "进行中"
    assert item["preview"] == "订单1001到哪了" and item["has_summary"] is True


async def test_conversation_messages_returns_items(client, monkeypatch):
    seen = {}

    async def fake_get(cid):
        seen["cid"] = cid
        return object()                                     # 存在即可

    async def fake_dialog(cid):
        return [
            type("M", (), {"role": "user", "content": "订单1001到哪了",
                           "created_at": datetime(2026, 9, 18, 10, 0, 0)})(),
            type("M", (), {"role": "assistant", "content": "在路上",
                           "created_at": None})(),
        ]

    monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
    monkeypatch.setattr(conversations.repository, "list_dialog_messages", fake_dialog)
    r = await client.get("/api/conversations/7/messages")
    assert r.status_code == 200
    assert seen["cid"] == 7
    items = r.json()["items"]
    assert [i["role"] for i in items] == ["user", "assistant"]
    assert items[0]["content"] == "订单1001到哪了"
    assert items[0]["created_at"] == "2026-09-18T10:00:00"
    assert items[1]["created_at"] is None                   # None 不炸序列化


async def test_conversation_messages_404_when_missing(client, monkeypatch):
    async def fake_get(cid): return None
    monkeypatch.setattr(conversations.repository, "get_conversation", fake_get)
    r = await client.get("/api/conversations/999/messages")
    assert r.status_code == 404
