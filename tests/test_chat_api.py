# tests/test_chat_api.py — ch05 契约:/api/chat 走图 astream(runtime.stream_turn),SSE 多 actions 帧
import json

import httpx
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.main import app


def make_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def read_sse(client, payload: dict):
    """返回 (事件列表, 是否出现 [DONE], 原始行)。事件不含 [DONE] 终止帧。"""
    events, done, lines = [], False, []
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            lines.append(line)
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):]
            if payload_str == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload_str))
    return events, done, lines


def _patch_stream(monkeypatch, events):
    async def fake_stream(user_id, message, conversation_id):
        for e in events:
            yield e
    monkeypatch.setattr(runtime, "stream_turn", fake_stream)


async def test_chat_stream_frame_sequence_with_actions(monkeypatch):
    _patch_stream(monkeypatch, [
        {"type": "tool", "name": "query_logistics"},
        {"type": "delta", "text": "您的"},
        {"type": "delta", "text": "订单运输中"},
        {"type": "citations", "items": [{"n": 1}]},
        {"type": "actions", "items": [{"type": "transfer_human"},
                                      {"type": "create_ticket", "draft": {}}]},
        {"type": "done", "conversation_id": 5},
    ])
    client = make_client()
    events, done, lines = await read_sse(client, {"user_id": "u1", "message": "订单 1001 到哪了"})
    assert '"event": "actions"' in "".join(lines)
    assert '"event": "tool", "name": "query_logistics"' in "".join(lines)
    deltas = [e["delta"] for e in events if "delta" in e]
    assert "".join(deltas) == "您的订单运输中"
    assert [e for e in events if e.get("event") == "citations"] == [{"event": "citations", "items": [{"n": 1}]}]
    assert events[-2] == {"event": "actions", "items": [{"type": "transfer_human"},
                                                        {"type": "create_ticket", "draft": {}}]}
    done_ev = [e for e in events if e.get("event") == "done"]
    assert done_ev == [{"event": "done", "conversation_id": 5}]
    assert done


async def test_chat_stream_error_on_conversation_not_found(monkeypatch):
    async def fake_stream(*a, **k):
        raise runtime.ConversationNotFound(999)
        yield  # pragma: no cover
    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    client = make_client()
    events, done, lines = await read_sse(client, {"user_id": "u1", "message": "hi",
                                                  "conversation_id": 999})
    err = [e for e in events if "message" in e]
    assert err and err[0]["message"] == "会话不存在"
    assert "data: [DONE]" not in lines            # 错误流不带终止帧


async def test_chat_stream_error_on_db_error(monkeypatch):
    async def fake_stream(*a, **k):
        raise SQLAlchemyError("db down")
        yield  # pragma: no cover
    monkeypatch.setattr(runtime, "stream_turn", fake_stream)
    client = make_client()
    events, done, _ = await read_sse(client, {"user_id": "u1", "message": "hi"})
    assert events[0]["message"] == "数据库暂时不可用,请稍后重试"


async def test_chat_empty_message_422():
    client = make_client()
    resp = await client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422                # 未进流
