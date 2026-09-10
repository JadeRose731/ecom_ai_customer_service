# tests/test_actions_api.py — ch05 建工单按钮端点(用户点了才写库)
from app.api import actions


async def test_create_ticket_action_writes(client, monkeypatch):
    async def fake_create(cid, desc, ttype):
        assert cid == 5 and ttype == "投诉"
        return "T20260715001"

    monkeypatch.setattr(actions.repository, "create_ticket", fake_create)
    r = await client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "东西坏了", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"] == "T20260715001"
    assert r.json()["status"] == "已转人工"


async def test_create_ticket_action_rejects_bad_type(client):
    r = await client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "x", "ticket_type": "乱填"})
    assert r.status_code == 422


async def test_create_ticket_action_rejects_blank_description(client):
    r = await client.post("/api/actions/create-ticket", json={
        "conversation_id": 5, "description": "", "ticket_type": "售后"})
    assert r.status_code == 422
