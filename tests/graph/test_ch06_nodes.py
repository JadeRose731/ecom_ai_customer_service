import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({"messages": [HumanMessage("这个能退吗")],
                                       "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"                 # 退款退货 → refund_flow
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83     # 不覆盖 confidence_check 的 confidence 键


def test_get_chat_model_honors_model_override():
    from app.core.llm import get_chat_model
    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    d = get_chat_model()
    from app.config import settings
    assert d.model_name == settings.chat_model


@pytest.mark.asyncio
async def test_resolve_reference_passthrough_when_complete(monkeypatch):
    async def fake_resolve(q, history=""):
        return q  # 已完整,原样
    monkeypatch.setattr(nodes.coref, "resolve", fake_resolve)
    out = await nodes.resolve_reference({"messages": [HumanMessage("蓝牙耳机的保修期多久")]})
    assert out["resolved_query"] == "蓝牙耳机的保修期多久"
    assert out["trace"]["coref"] == "passthrough"


@pytest.mark.asyncio
async def test_resolve_reference_rewrites_with_history(monkeypatch):
    async def fake_resolve(q, history=""):
        return "蓝牙耳机还能申请退货吗"
    monkeypatch.setattr(nodes.coref, "resolve", fake_resolve)
    out = await nodes.resolve_reference({"messages": [
        HumanMessage("蓝牙耳机什么时候到"), AIMessage("预计明天"), HumanMessage("这个能退吗")]})
    assert out["resolved_query"] == "蓝牙耳机还能申请退货吗"
    assert out["trace"]["coref"] == "rewrite"


def test_extract_order_id():
    assert nodes._extract_order_id("订单1001的物流") == "1001"
    assert nodes._extract_order_id("尾号 20260701 那单") == "20260701"
    assert nodes._extract_order_id("我要退货") is None


@pytest.mark.asyncio
async def test_fetch_order_uses_id_in_query(monkeypatch):
    out = await nodes.fetch_order({"resolved_query": "订单1001能退吗", "user_id": "u1"})
    assert out["order_id"] == "1001"
    assert out["order_data"]["order_id"] == "1001"          # order_snapshot 同源
    assert out["trace"]["fetch_order"]["order_id"] == "1001"


@pytest.mark.asyncio
async def test_fetch_order_interrupts_when_missing(monkeypatch):
    # interrupt() 只能在编译图内跑(Task 1 冒烟 D:图外直接调是 RuntimeError),
    # 故缺单路径经最小编译图 + InMemorySaver ainvoke 测真实中断 surface。
    monkeypatch.setattr(nodes.business, "list_user_orders",
                        lambda uid: [{"order_id": "1001", "product": "猫粮", "status": "已签收", "amount": 99}])
    from langgraph.graph import START, END, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from app.graph.state import ConversationState
    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", END)
    graph = b.compile(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"resolved_query": "我要退款", "user_id": "u1"},
                              {"configurable": {"thread_id": "t-fetch"}})
    payload = out["__interrupt__"][0].value      # Task 1 冒烟 A/C 钉死此取法
    assert payload["type"] == "select_order"
    assert payload["orders"][0]["order_id"] == "1001"


def test_list_user_orders_stable_and_queryable():
    from app.tools import business
    a = business.list_user_orders("u-42")
    b = business.list_user_orders("u-42")
    assert a == b and len(a) >= 2                            # 同 user 稳定
    # 选中即可 query_order(同源快照)
    snap = business.order_snapshot(a[0]["order_id"])
    assert snap["order_id"] == a[0]["order_id"] and snap["product"] == a[0]["product"]


@pytest.mark.asyncio
async def test_retrieve_policy_expands_dedups_merges(monkeypatch):
    async def fake_expand(q):
        return ["退货政策", "无理由退换货", "退货时限"]
    # 三条 query 各自的召回:id=1 在两条里出现(取高分),id=2 只在一条
    per_query = {
        "退货政策": [{"id": 1, "question": "退货", "answer": "7天无理由", "rerank_score": 0.7,
                     "section_path": "政策/退货", "content_type": "policy"}],
        "无理由退换货": [{"id": 1, "question": "退货", "answer": "7天无理由", "rerank_score": 0.9,
                       "section_path": "政策/退货", "content_type": "policy"},
                      {"id": 2, "question": "运费", "answer": "质量问题商家承担", "rerank_score": 0.6,
                       "section_path": "政策/运费", "content_type": "policy"}],
        "退货时限": [],
    }
    async def fake_search(q, **k):
        return per_query.get(q, [])
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)
    monkeypatch.setattr(nodes.retrieval, "arrange_head_tail", lambda h: h)

    out = await nodes.retrieve_policy({"resolved_query": "这个订单能退吗",
                                       "order_data": {"status": "已签收"}})
    # 去重:id=1 只保留一次(取 0.9),id=2 保留;共 2 条
    assert len(out["citations"]) == 2
    ids = [c["id"] for c in out["citations"]]
    assert ids == [1, 2]                              # 按 rerank_score 降序(0.9, 0.6)
    assert "[1]" in out["evidence"] and "[2]" in out["evidence"]
    assert out["trace"]["retrieve_policy"]["hits"] == 2
    assert out["trace"]["retrieve_policy"]["queries"] == ["退货政策", "无理由退换货", "退货时限"]


@pytest.mark.asyncio
async def test_agent_tools_intercepts_submit_refund():
    ai = AIMessage(content="", tool_calls=[
        {"id": "r1", "name": "submit_refund", "args": {"order_id": "1001", "reason": None}}])
    out = await nodes.agent_tools({"messages": [ai]})
    assert out["suggested_actions"] == [{"type": "refund_form", "draft": {"order_id": "1001", "reason": None}}]
    tm = out["messages"][0]
    assert tm.name == "submit_refund"                       # 合成 ToolMessage 促收敛
    assert "退款" in tm.content


def test_agent_messages_injects_order_and_policy_on_refund():
    msgs = nodes._agent_messages({
        "route": "refund_flow",
        "order_data": {"order_id": "1001", "status": "已签收", "product": "猫粮 5kg"},
        "evidence": "[1] 退货: 7天无理由",
        "messages": [HumanMessage("这单能退吗")]})
    sys = msgs[0]
    assert isinstance(sys, SystemMessage)
    assert "7天无理由" in sys.content                        # 政策证据注入
    assert "猫粮 5kg" in sys.content and "submit_refund" in sys.content  # 订单数据 + 判定指令


def test_agent_messages_knowledge_path_still_injects_evidence():
    msgs = nodes._agent_messages({
        "route": "knowledge", "evidence": "[1] 运费: 满99包邮",
        "messages": [HumanMessage("运费多少")]})
    assert "满99包邮" in msgs[0].content                     # 放宽条件后知识路不回归
