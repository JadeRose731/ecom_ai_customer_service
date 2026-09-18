# tests/test_refund_api.py — ch06 退款表单提交端点(复用 tickets 表,ticket_type='退款')
import json

from app.api import actions


async def test_create_refund_writes_ticket(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 7 and ttype == "退款" and "1001" in desc and "质量问题" in desc
        return "T20260716001"

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = await client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "质量问题"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260716001"
    assert r.json()["status"] == "退款申请已提交"


async def test_create_refund_rejects_bad_reason(client):
    r = await client.post("/api/actions/create-refund", json={
        "conversation_id": 7, "order_id": "1001", "reason": "乱填"})
    assert r.status_code == 422


async def test_resume_endpoint_streams(client, monkeypatch):
    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == "1001"
        # R15:interrupt 事件带 conversation_id,端点帧应透传(契约与 /api/chat 同构)
        yield {"type": "interrupt", "kind": "select_order",
               "orders": [{"order_id": "1001"}], "conversation_id": 3}
        yield {"type": "delta", "text": "这一单可以退款"}
        yield {"type": "actions", "items": [{"type": "refund_form", "draft": {"order_id": "1001"}}]}
        yield {"type": "done", "conversation_id": 3}
    from app.api import actions
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = await client.post("/api/actions/resume", json={"conversation_id": 3, "order_id": "1001"})
    assert r.status_code == 200
    body = r.text
    assert "这一单可以退款" in body
    assert "refund_form" in body and "[DONE]" in body
    # R15:interrupt SSE 帧含 conversation_id(解析验证,不依赖 json.dumps 分隔符)
    intr_frames = [ln[len("data: "):] for ln in body.splitlines()
                   if ln.startswith("data: ") and '"interrupt"' in ln]
    assert intr_frames, "body 缺 interrupt 帧"
    payload = json.loads(intr_frames[0])
    assert payload["kind"] == "select_order"
    assert payload["conversation_id"] == 3
    assert payload["orders"] == [{"order_id": "1001"}]


async def test_resume_endpoint_confirmed_branch(client, monkeypatch):
    """ch08:工单预览确认走同一端点,resume 值为 {"confirmed": bool}(graph 按 confirmed 分派)。"""
    from app.api import actions

    async def fake_stream_resume(cid, resume_value):
        assert cid == 3 and resume_value == {"confirmed": True}
        yield {"type": "delta", "text": "T20260717001"}
        yield {"type": "done", "conversation_id": 3}
    monkeypatch.setattr(actions.runtime, "stream_resume", fake_stream_resume)
    r = await client.post("/api/actions/resume", json={"conversation_id": 3, "confirmed": True})
    assert r.status_code == 200
    assert "T20260717001" in r.text and "[DONE]" in r.text


async def test_resume_endpoint_requires_one_of_fields(client):
    r = await client.post("/api/actions/resume", json={"conversation_id": 3})
    assert r.status_code == 400                   # order_id 与 confirmed 至少传一个
