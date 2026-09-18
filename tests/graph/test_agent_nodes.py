import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph import nodes


def test_agent_messages_injects_evidence_on_knowledge():
    msgs = nodes._agent_messages(
        {"route": "knowledge", "evidence": "[1] 退货: 7天",
         "messages": [HumanMessage("能退吗")]})
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == AGENT_SYSTEM          # ch07:system 恒为静态人设,证据挪用户侧
    ctx = msgs[-1]                                   # 材料消息插在用户那句之后
    assert "[1] 退货: 7天" in ctx.content
    assert "query_faq" in ctx.content  # 指示别再检索


def test_agent_messages_no_evidence_on_business():
    msgs = nodes._agent_messages({"route": "business", "messages": [HumanMessage("订单1001")]})
    assert isinstance(msgs[0], SystemMessage)
    assert "已检索到的知识证据" not in msgs[0].content


def _patch_specs(monkeypatch):
    """agent_llm/agent_tools 都会现拉全量清单;单测一律打桩——真拉会连 MCP Server
    (8101/8102 没起时 wait_for+冷却也要好几秒,且测试不该依赖外部进程)。"""
    async def fake_specs():
        return []
    monkeypatch.setattr(nodes.registry, "get_all_specs", fake_specs)


@pytest.mark.asyncio
async def test_agent_llm_accumulates_steps_and_tokens(monkeypatch):
    _patch_specs(monkeypatch)
    ai = AIMessage("好的", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

    class FakeModel:
        def bind_tools(self, tools):
            return self
        async def ainvoke(self, msgs, config=None):
            return ai

    monkeypatch.setattr(nodes, "get_chat_model", lambda **k: FakeModel())
    out = await nodes.agent_llm({"messages": [HumanMessage("hi")], "steps": 1, "tokens_used": 100})
    assert out["steps"] == 2
    assert out["tokens_used"] == 115
    assert out["messages"][0] is ai


@pytest.mark.asyncio
async def test_agent_tools_executes_normal_tool(monkeypatch):
    from app.tools.engine import ToolRun
    from langchain_core.messages import ToolMessage

    _patch_specs(monkeypatch)

    async def fake_exec(tc, cid, specs):
        return ToolRun(tool_call_id=tc["id"], name=tc["name"], ok=True, status="成功",
                       tool_message=ToolMessage(content='{"status":"已发货"}', tool_call_id=tc["id"], name=tc["name"]))

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "t1"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    assert out["messages"][0].name == "query_logistics"
    assert not out.get("suggested_actions")


@pytest.mark.asyncio
async def test_agent_tools_intercepts_create_ticket(monkeypatch):
    _patch_specs(monkeypatch)

    async def fake_exec(tc, cid, specs):
        raise AssertionError("create_ticket 不应被执行(应拦截为提议)")

    monkeypatch.setattr(nodes.engine, "execute_tool_call", fake_exec)
    ai = AIMessage("", tool_calls=[{"name": "create_ticket",
                    "args": {"description": "屏幕碎了", "ticket_type": "售后"}, "id": "t9"}])
    out = await nodes.agent_tools({"messages": [ai], "conversation_id": 5})
    act = out["suggested_actions"][0]
    assert act["type"] == "create_ticket"
    assert act["draft"]["description"] == "屏幕碎了"
    # 回一条合成 ToolMessage 让模型收敛(不再调工具)
    assert out["messages"][0].tool_call_id == "t9"
