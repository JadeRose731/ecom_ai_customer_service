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


@pytest.mark.asyncio
async def test_stream_turn_emits_interrupt_event(monkeypatch):
    async def fake_create(uid): return 9
    async def fake_append(cid, role, content=None, **k): return 1
    monkeypatch.setattr(runtime.repository, "create_conversation", fake_create)
    monkeypatch.setattr(runtime.repository, "append_message", fake_append)

    class Intr:
        def __init__(self, value): self.value = value

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            yield ("updates", {"fetch_order": None})
            yield ("updates", {"__interrupt__": (Intr({"type": "select_order",
                     "orders": [{"order_id": "1001"}]}),)})
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_turn("u1", "我要退款", None)]
    intr = [e for e in events if e["type"] == "interrupt"]
    assert intr and intr[0]["kind"] == "select_order"
    assert intr[0]["orders"] == [{"order_id": "1001"}]


@pytest.mark.asyncio
async def test_resume_turn_drives_command(monkeypatch):
    from langgraph.types import Command
    seen = {}

    class FakeGraph:
        async def ainvoke(self, inp, config):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            return {"answer": "这一单可以退款", "messages": []}
    async def fake_get(cid): return object()
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    out = await runtime.resume_turn(5, "1001")
    assert seen["is_command"] and seen["resume"] == "1001"
    assert out["conversation_id"] == 5 and out["state"]["answer"] == "这一单可以退款"


@pytest.mark.asyncio
async def test_stream_resume_missing_conversation_raises(monkeypatch):
    async def fake_get(cid): return None
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    with pytest.raises(runtime.ConversationNotFound):
        async for _ in runtime.stream_resume(999, "1001"):
            pass


@pytest.mark.asyncio
async def test_stream_resume_passes_command(monkeypatch):
    from langchain_core.messages import AIMessage
    from langgraph.types import Command
    seen = {}

    class FakeGraph:
        async def astream(self, inp, config, stream_mode=None):
            seen["is_command"] = isinstance(inp, Command)
            seen["resume"] = getattr(inp, "resume", None)
            yield ("messages", (AIMessage("这一单可以退款"), {"langgraph_node": "agent_llm"}))

    async def fake_get(cid): return object()
    monkeypatch.setattr(runtime.repository, "get_conversation", fake_get)
    monkeypatch.setattr(runtime, "get_graph", lambda: FakeGraph())
    events = [e async for e in runtime.stream_resume(5, "1001")]
    assert seen["is_command"] and seen["resume"] == "1001"
    deltas = [e["text"] for e in events if e["type"] == "delta"]
    assert "这一单可以退款" in deltas
    assert events[-1]["type"] == "done" and events[-1]["conversation_id"] == 5


def test_dedup_actions_keep_first():
    a1 = {"type": "create_ticket", "draft": {"title": "查物流"}}
    a2 = {"type": "create_ticket", "draft": {"title": "查物流"}}
    a3 = {"type": "create_ticket", "draft": {"title": "申请换货"}}
    out = runtime.dedup_actions([a1, a2, a3])
    assert out == [a1, a3]  # 同 type+同 draft 去重留首条;draft 不同保留


def test_graph_input_resets_per_turn_output_channels():
    """入口把上一轮输出通道全部清零,防跨轮泄漏(ch05 C1)。"""
    inp = runtime._graph_input("u1", "你好", 42)
    # ch05 输出通道
    assert inp["intent"] == ""
    assert inp["route"] == ""
    assert inp["evidence"] == ""
    assert inp["citations"] == []
    assert inp["evidence_strong"] is False
    assert inp["answer"] == ""
    assert inp["suggested_actions"] == []
    assert inp["steps"] == 0 and inp["tokens_used"] == 0
    # ch06 四个无 reducer 标量
    assert inp["resolved_query"] == "" and inp["intent_confidence"] == 0.0
    assert inp["order_id"] == "" and inp["order_data"] == {}
