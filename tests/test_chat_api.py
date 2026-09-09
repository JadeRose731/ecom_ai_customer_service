# tests/test_chat_api.py — ch02 新契约:/api/chat 升级为带工具链的 SSE 流式主入口
import json

import httpx
import pytest
from langchain_core.messages import AIMessage

from app.api import chat as chat_api
from app.db import repository as repo
from app.main import app
from tests.test_agent_orchestration import FakeModel

def make_client(model: FakeModel) -> httpx.AsyncClient:
    app.dependency_overrides[chat_api.get_model] = lambda: model
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

async def read_sse(client, payload: dict):
    """返回 (事件列表, 是否出现 [DONE])。事件不含 [DONE] 终止帧。"""
    events, done = [], False
    async with client.stream("POST", "/api/chat", json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload_str = line[len("data: "):]
            if payload_str == "[DONE]":
                done = True
            else:
                events.append(json.loads(payload_str))
    return events, done

def teardown_function():
    app.dependency_overrides.clear()

async def test_chat_with_tools_streams_frame_sequence(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的", "订单", "运输中"])
    client = make_client(model)
    events, done = await read_sse(client, {"user_id": "u1", "message": "订单 1001 到哪了"})
    assert events[0] == {"event": "tool", "name": "query_logistics"}
    deltas = [e["delta"] for e in events if "delta" in e]
    assert "".join(deltas) == "您的订单运输中"
    done_ev = [e for e in events if e.get("event") == "done"]
    assert len(done_ev) == 1 and isinstance(done_ev[0]["conversation_id"], int)
    assert done
    msgs = await repo.list_messages(done_ev[0]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]

async def test_chat_no_tool_direct_answer(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~")])
    client = make_client(model)
    events, done = await read_sse(client, {"user_id": "u1", "message": "你好"})
    assert all("tool" not in e for e in events)
    deltas = [e["delta"] for e in events if "delta" in e]
    assert "".join(deltas) == "你好,喵~"
    assert done
    done_ev = [e for e in events if e.get("event") == "done"]
    msgs = await repo.list_messages(done_ev[0]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant"]

async def test_chat_unknown_conversation_emits_error_no_done(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    client = make_client(model)
    lines = []
    async with client.stream("POST", "/api/chat",
                             json={"user_id": "u1", "message": "hi", "conversation_id": 999999}) as resp:
        async for line in resp.aiter_lines():
            lines.append(line)
    assert "event: error" in lines
    err = [json.loads(l[len("data: "):]) for l in lines if l.startswith("data: ") and l != "data: [DONE]"]
    assert err and err[0]["message"] == "会话不存在"
    assert "data: [DONE]" not in lines            # 错误流不带终止帧

async def test_chat_empty_message_422(db_session_factory, db_clean):
    client = make_client(FakeModel([AIMessage(content="x")]))
    resp = await client.post("/api/chat", json={"user_id": "u1", "message": ""})
    assert resp.status_code == 422                # 未进流
