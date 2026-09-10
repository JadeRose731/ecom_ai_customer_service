# tests/test_agent_api.py
# ch05:/api/agent 改走图 ainvoke(runtime.run_turn),断言基线从旧编排改为「经图返回 answer + tool_calls」。
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy.exc import SQLAlchemyError

from app.graph import runtime
from app.main import app


def _fake_state(answer="订单 1001 正在运输中。"):
    ai = AIMessage("", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}])
    tm = ToolMessage(content='{"status":"运输中"}', tool_call_id="c1", name="query_logistics")
    return {
        "messages": [HumanMessage("订单 1001 到哪了"), ai, tm, AIMessage(answer)],
        "intent": "物流", "route": "business", "suggested_actions": [],
    }


def test_agent_endpoint_returns_tool_trace(monkeypatch):
    async def fake_run(user_id, message, conversation_id):
        return {"conversation_id": 12, "state": _fake_state()}
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "订单 1001 到哪了"})
    assert r.status_code == 200
    body = r.json()
    assert body["conversation_id"] == 12
    assert body["answer"] == "订单 1001 正在运输中。"
    assert body["tool_calls"][0]["name"] == "query_logistics"
    assert body["tool_results"][0]["ok"] is True


def test_agent_endpoint_chitchat_no_tools(monkeypatch):
    async def fake_run(user_id, message, conversation_id):
        return {"conversation_id": 3,
                "state": {"messages": [HumanMessage("你好"), AIMessage("你好呀")],
                          "intent": "闲聊", "route": "chitchat", "answer": "你好呀",
                          "suggested_actions": []}}
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "你好"})
    assert r.status_code == 200
    body = r.json()
    assert body["tool_calls"] == []
    assert body["answer"] == "你好呀"


def test_agent_endpoint_carries_suggested_actions(monkeypatch):
    async def fake_run(user_id, message, conversation_id):
        state = _fake_state()
        state["suggested_actions"] = [{"type": "transfer_human"}]
        return {"conversation_id": 12, "state": state}
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.json()["suggested_actions"] == [{"type": "transfer_human"}]


def test_agent_endpoint_404_on_unknown_conversation(monkeypatch):
    async def fake_run(*a, **k):
        raise runtime.ConversationNotFound(999)
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi", "conversation_id": 999})
    assert r.status_code == 404


def test_agent_endpoint_503_on_db_error(monkeypatch):
    async def fake_run(*a, **k):
        raise SQLAlchemyError("db down")
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 503


def test_agent_endpoint_502_on_upstream_error(monkeypatch):
    async def fake_run(*a, **k):
        raise RuntimeError("model blew up")
    monkeypatch.setattr(runtime, "run_turn", fake_run)
    client = TestClient(app)
    r = client.post("/api/agent", json={"user_id": "u1", "message": "hi"})
    assert r.status_code == 502


def test_agent_endpoint_422_on_missing_fields():
    client = TestClient(app)
    assert client.post("/api/agent", json={"message": "hi"}).status_code == 422
