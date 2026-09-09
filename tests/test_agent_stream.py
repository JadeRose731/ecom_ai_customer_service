# tests/test_agent_stream.py
import pytest
from langchain_core.messages import AIMessage

from app.core import agent
from app.db import repository as repo
from tests.test_agent_orchestration import FakeModel

async def test_stream_with_tools_emits_tool_delta_done(db_session_factory, db_clean):
    first = AIMessage(content="", tool_calls=[
        {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    model = FakeModel([first], stream_tokens=["您的", "订单", "运输中"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "订单 1001 到哪了", None, model=model)]
    assert events[0] == {"type": "tool", "name": "query_logistics"}
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "".join(deltas) == "您的订单运输中"
    assert events[-1]["type"] == "done" and isinstance(events[-1]["conversation_id"], int)
    assert model.bind_calls == 1                       # astream 收敛不 bind
    msgs = await repo.list_messages(events[-1]["conversation_id"])
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]

async def test_stream_no_tool_direct_delta_no_astream(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="你好,喵~")])
    events = [ev async for ev in agent.stream_agent_turn("u1", "你好", None, model=model)]
    assert all(e["type"] != "tool" for e in events)
    assert {"type": "delta", "text": "你好,喵~"} in events
    assert events[-1]["type"] == "done"
    assert model.astream_messages == []                # 无工具:turn1 即答案,不走 astream
    cid = events[-1]["conversation_id"]
    msgs = await repo.list_messages(cid)
    assert [m.role for m in msgs] == ["user", "assistant"]

async def test_stream_unknown_conversation_raises(db_session_factory, db_clean):
    model = FakeModel([AIMessage(content="hi")])
    agen = agent.stream_agent_turn("u1", "hi", 999999, model=model)
    with pytest.raises(agent.ConversationNotFound):
        async for _ in agen:
            pass
