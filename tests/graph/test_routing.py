from langchain_core.messages import AIMessage

from app.config import settings
from app.graph.routing import (
    confidence_gate, route_by_intent, should_continue,
)


def test_route_maps_seven_to_four():
    cases = {
        "商品咨询": "knowledge", "退款退货": "knowledge",
        "物流": "business", "订单": "business", "售后": "business",
        "投诉": "complaint", "闲聊": "chitchat",
    }
    for intent, route in cases.items():
        assert route_by_intent({"intent": intent}) == route


def test_route_unknown_intent_defaults_business():
    assert route_by_intent({"intent": "火星语"}) == "business"


def test_confidence_gate():
    assert confidence_gate({"evidence_strong": True}) == "strong"
    assert confidence_gate({"evidence_strong": False}) == "weak"
    assert confidence_gate({}) == "weak"


def test_should_continue_stops_when_no_tool_calls():
    state = {"messages": [AIMessage("答完了")], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_should_continue_continues_on_tool_calls_under_budget():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 0}
    assert should_continue(state) == "continue"


def test_should_continue_stops_on_step_cap():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": settings.max_agent_steps, "tokens_used": 0}
    assert should_continue(state) == "stop"


def test_token_花销不再当停止条件():
    # 环里只按步数封顶。token 记账继续走(进 trace、给 ch09 统计),但不参与判断
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {}, "id": "1"}])
    state = {"messages": [ai], "steps": 1, "tokens_used": 10_000_000}
    assert should_continue(state) == "continue"
