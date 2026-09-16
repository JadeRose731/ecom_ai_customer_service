# tests/test_refund_api.py — ch06 退款表单提交端点(复用 tickets 表,ticket_type='退款')
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
