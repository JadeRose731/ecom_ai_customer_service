from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.agent import _views_from_state


def test_views_rebuilt_from_messages():
    ai = AIMessage("", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"已发货"}', tool_call_id="c1", name="query_order")
    calls, results = _views_from_state({"messages": [HumanMessage("q"), ai, tm, AIMessage("答")]})
    assert calls[0].name == "query_order"
    assert results[0].tool_call_id == "c1"
    assert results[0].name == "query_order"
