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


@pytest.mark.asyncio
async def test_pending_thread_new_dict_input_reinterrupts():
    """I1b(终审钉死):挂起线程收全新 dict 输入的 langgraph 现状语义,防升级无声变更。

    开发期探针实测(langgraph 本仓锁定版):全新 dict 输入 = **整图从 START 头重跑**
    (上游 A 重跑一次、B 从头重跑再次暂停);Command(resume) 才是「只重跑被中断节点 B、
    上游 A 不动」。终审 brief 预期的「A 计数仍=1(上游未重跑)」与实测不符,按测试目的
    (钉死现状)如实断言实测语义;前端守卫 I1a 拦截的正是这条新输入重跑路。
    """
    from typing_extensions import TypedDict

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    class MinState(TypedDict, total=False):
        marker: str
        count: int

    calls = {"a": 0, "b": 0}      # 真实执行次数(闭包计数,不受输入重置影响)

    async def node_a(state):
        calls["a"] += 1           # 计数节点 A:写 marker 且计数,不 interrupt
        return {"marker": "A-done", "count": state.get("count", 0) + 1}

    async def node_b(state):
        calls["b"] += 1
        oid = interrupt({"type": "select_order"})
        return {"marker": f"B:{oid}"}

    b = StateGraph(MinState)
    b.add_node("a", node_a)
    b.add_node("b", node_b)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)
    graph = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-pending-reinterrupt"}}

    out1 = await graph.ainvoke({"count": 0, "marker": ""}, cfg)
    assert "__interrupt__" in out1
    assert out1["__interrupt__"][0].value == {"type": "select_order"}
    assert out1["count"] == 1 and calls["a"] == 1            # A 恰好跑一次

    # 同 config、全新 dict 输入:现状=从 START 整图重跑,A/B 各再跑一次,再次暂停
    out2 = await graph.ainvoke({"count": 0, "marker": ""}, cfg)
    assert "__interrupt__" in out2
    assert out2["__interrupt__"][0].value == {"type": "select_order"}
    assert calls["a"] == 2 and calls["b"] == 2               # 上游 A 被重跑(现状如实钉死)
    assert out2["count"] == 1                                # 输入重置后 A 本轮跑一次

    # Command(resume) 才是「只重跑被中断节点」:B 再跑一次(interrupt 返回回填值),A 不动
    out3 = await graph.ainvoke(Command(resume="1001"), cfg)
    assert "__interrupt__" not in out3
    assert out3["marker"] == "B:1001"
    assert calls["a"] == 2 and calls["b"] == 3


@pytest.mark.asyncio
async def test_resume_reruns_read_once(monkeypatch):
    """I2(spec §17 R2):缺单 interrupt → resume 续跑,「同一有效查单」不重复副作用。

    fetch_order 的 interrupt 前代码只读(list_user_orders/order_snapshot 均无写库),
    resume 时节点从 fetch_order 头重跑:mock 列表重生成一遍(同 user 同种子同 id,只读),
    用户所选单的 order_snapshot 在整个 interrupt→resume 周期恰好查一次。
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from app.graph.state import ConversationState

    # 打补丁前先取未包装的 mock 列表:同 user 稳定,算出列表期的快照调用序列
    list_before = nodes.business.list_user_orders("u1")
    list_ids = [d["order_id"] for d in list_before]

    real_snapshot = nodes.business.order_snapshot
    snap_calls: list[str] = []

    def counting_snapshot(oid):
        snap_calls.append(str(oid))
        return real_snapshot(oid)                 # 查有单 id 返回真 snapshot dict

    monkeypatch.setattr(nodes.business, "order_snapshot", counting_snapshot)

    async def fake_expand(q):
        return [q]                                # 防上游扩写分叉,seed 原样透传
    monkeypatch.setattr(nodes.query_understanding, "expand_queries", fake_expand)

    async def fake_search(q, **k):
        return []                                 # 检索不打真库
    monkeypatch.setattr(nodes.retrieval, "search_knowledge", fake_search)

    b = StateGraph(ConversationState)
    b.add_node("fetch_order", nodes.fetch_order)
    b.add_node("retrieve_policy", nodes.retrieve_policy)
    b.add_edge(START, "fetch_order")
    b.add_edge("fetch_order", "retrieve_policy")
    b.add_edge("retrieve_policy", END)
    graph = b.compile(checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t-resume-read-once"}}

    # 无订单号问退款 → interrupt(所选单尚未查)
    out1 = await graph.ainvoke({"messages": [HumanMessage("我要退款")],
                                "resolved_query": "我要退款", "user_id": "u1",
                                "order_id": "", "order_data": {}}, cfg)
    assert "__interrupt__" in out1
    assert [o["order_id"] for o in out1["__interrupt__"][0].value["orders"]] == list_ids
    # spec R2「interrupt 前只读」:首轮仅 mock 列表各快照一次,不触碰任何「所选单」
    assert snap_calls == list_ids

    # resume 回填 1001 → fetch_order 从头重跑 → retrieve_policy → END
    out2 = await graph.ainvoke(Command(resume="1001"), cfg)
    assert "__interrupt__" not in out2
    assert out2["order_id"] == "1001"
    assert out2["order_data"]["order_id"] == "1001"
    # 精确轨迹(读 fetch_order 源码钉死):首轮列表 N 次 + resume 重跑列表 N 次(同 id、
    # 只读)+ 所选单恰查 1 次 —— spec R2「缺单→interrupt→resume→query_order 只一次」
    assert snap_calls == [*list_ids, *list_ids, "1001"]
    assert snap_calls.count("1001") == 1


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


@pytest.mark.asyncio
async def test_script_reply_by_intent():
    from app.core.prompts import SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER
    chit = await nodes.script_reply({"intent": "闲聊"})
    assert chit["answer"] == SCRIPT_REPLY_CHITCHAT
    assert chit["trace"]["route"] == "fallback_script"
    other = await nodes.script_reply({"intent": "其他"})
    assert other["answer"] == SCRIPT_REPLY_OTHER
