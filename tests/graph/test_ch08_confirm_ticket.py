"""graph 级:建工单确认流。业务路由 → main_agent 发 create_ticket → agent_tools 顶置 interrupt
推预览 → resume confirmed 两分支(true 落库回工单号 / false 权限拒绝审计+不落库)。"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.graph import nodes
from app.graph.build import build_graph
from app.tools import engine, registry


class FakeModel:
    """第一次调用发 create_ticket tool_call,之后回普通文本收敛。"""
    def __init__(self):
        self.calls = 0
    def bind_tools(self, tools): return self
    def bind(self, **kw): return self
    async def ainvoke(self, msgs, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(content="", tool_calls=[{
                "name": "create_ticket", "id": "tc-1",
                "args": {"description": "猫砂盆漏电", "ticket_type": "售后"}}])
        return AIMessage(content="已为您创建工单")


async def fake_coref(q, h): return q


@pytest.fixture()
def wired(monkeypatch):
    # 骨架原写「售后」→ 路由 refund_flow 会先弹 select_order 中断,轮不到工单预览卡;
    # 本测试要的是直达 main_agent 的业务路,intent 用「订单」(断言语义不变)
    async def fake_intent(q, h): return {"intent": "订单", "confidence": 0.95}
    monkeypatch.setattr(nodes.coref, "resolve", fake_coref)
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_intent)
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: FakeModel())
    async def only_builtin():
        return [s for s in registry.builtin_specs()]
    monkeypatch.setattr(nodes.registry, "get_all_specs", only_builtin)

    created, audits = [], []
    async def fake_create(cid, desc, ttype):
        created.append((cid, desc, ttype)); return "T20260717001"
    async def fake_audit(**kw): audits.append(kw)
    monkeypatch.setattr("app.db.repository.create_ticket", fake_create)
    monkeypatch.setattr(engine.repository, "insert_tool_audit", fake_audit)
    async def no_msg(*a, **kw): return 1
    monkeypatch.setattr(nodes.repository, "append_message", no_msg)
    return created, audits


def _input(text):
    from app.graph.runtime import _graph_input
    return _graph_input("u1", text, 1, 1, "", 0)


async def test_confirm_true_creates_ticket(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    st = await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    intr = st["__interrupt__"][0].value
    assert intr["type"] == "confirm_ticket"
    assert intr["preview"] == {"ticket_type": "售后", "description": "猫砂盆漏电"}
    assert created == []                                        # interrupt 时未执行

    st2 = await g.ainvoke(Command(resume={"confirmed": True}), cfg)
    assert created == [(1, "猫砂盆漏电", "售后")]
    assert any(a["status"] == "成功" for a in audits)
    assert "T20260717001" in str(st2["messages"])               # 工单号回灌到 ToolMessage


async def test_confirm_false_denied_and_audited(wired):
    created, audits = wired
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t2"}}
    await g.ainvoke(_input("帮我建个工单,猫砂盆漏电"), cfg)
    await g.ainvoke(Command(resume={"confirmed": False}), cfg)
    assert created == []                                        # 没建
    assert any(a["status"] == "权限拒绝" for a in audits)       # 审计落权限拒绝


async def test_missing_description_asks_instead_of_interrupt(wired, monkeypatch):
    """模型第一轮发缺 description 的 create_ticket → 校验拦下回灌(不 interrupt),
    第二轮模型收敛为追问文本。"""
    class NoDescModel(FakeModel):
        async def ainvoke(self, msgs, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_ticket", "id": "tc-1", "args": {"ticket_type": "咨询"}}])
            return AIMessage(content="请问您遇到的具体问题是什么呢?")
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: NoDescModel())
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}
    st = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st                            # 没弹卡
    created, audits = wired
    assert created == [] and any(a["status"] == "校验拦下" for a in audits)


async def test_empty_description_blocked_not_confirmed(wired, monkeypatch):
    """description="" 空串过不了 min_length=1 → 校验拦下回灌,不弹空预览卡(ch08 评审 Important#1:
    空描述预览卡对用户无意义,机械闸必须逼模型追问,不靠 prompt 自觉)。"""
    class EmptyDescModel(FakeModel):
        async def ainvoke(self, msgs, config=None):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="", tool_calls=[{
                    "name": "create_ticket", "id": "tc-1",
                    "args": {"description": "", "ticket_type": "咨询"}}])
            return AIMessage(content="请问您遇到的具体问题是什么呢?")
    monkeypatch.setattr(nodes, "get_chat_model", lambda **kw: EmptyDescModel())
    g = build_graph(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t4"}}
    st = await g.ainvoke(_input("帮我建个工单"), cfg)
    assert "__interrupt__" not in st                            # 没弹卡
    created, audits = wired
    assert created == [] and any(a["status"] == "校验拦下" for a in audits)
