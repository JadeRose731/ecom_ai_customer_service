import pytest

from app.graph import runtime


@pytest.mark.asyncio
async def test_run_turn_creates_conversation_when_none(monkeypatch):
    async def fake_create(uid): return 42
    async def fake_append(cid, role, content=None, **k): return 1

    class FakeGraph:
        async def ainvoke(self, inp, config):
            return {"answer": "hi", "messages": [], "conversation_id": inp["conversation_id"]}

    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())

    out = await runtime.run_turn("u1", "你好", None)
    assert out["conversation_id"] == 42


@pytest.mark.asyncio
async def test_run_turn_raises_when_conversation_missing(monkeypatch):
    async def fake_get(cid): return None
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        await runtime.run_turn("u1", "hi", 999)


@pytest.mark.asyncio
async def test_stream_turn_maps_events(monkeypatch):
    async def fake_create(uid): return 7
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    from langchain_core.messages import AIMessage, ToolMessage

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            # messages 模式:agent_llm 的 token
            yield ("messages", (AIMessage("已"), {"langgraph_node": "agent_llm"}))
            yield ("messages", (AIMessage("发货"), {"langgraph_node": "agent_llm"}))
            # messages 模式:非答复节点 token 应被过滤
            yield ("messages", (AIMessage("订单"), {"langgraph_node": "classify_intent"}))
            # updates 模式:工具帧 + citations + actions
            yield ("updates", {"agent_tools": {"messages": [
                ToolMessage(content="{}", tool_call_id="1", name="query_logistics")]}})
            yield ("updates", {"forced_rag": {"citations": [{"n": 1}]}})
            yield ("updates", {"complaint_reply": {"answer": "抱歉",
                    "suggested_actions": [{"type": "transfer_human"}]}})

    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "订单1001物流", None)]
    kinds = [e["type"] for e in events]
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    tools = [e["name"] for e in events if e["type"] == "tool"]
    assert "已" in deltas and "发货" in deltas
    assert "订单" not in deltas  # 非答复节点被过滤
    assert tools == ["query_logistics"]
    assert any(e["type"] == "citations" for e in events)
    assert any(e["type"] == "actions" for e in events)
    assert kinds[-1] == "done" and events[-1]["conversation_id"] == 7
