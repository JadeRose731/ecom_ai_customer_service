"""ch09 审核 API:列表/详情/通过(写回知识库)/驳回;只有待审可流转;写回失败状态不动。"""
import httpx
import pytest
from fastapi import FastAPI

from app.api import review as review_api
from app.db import repository


@pytest.fixture
async def client(db_session_factory, db_clean):
    # Plan 原稿用 TestClient + run_until_complete 造数(其备注也预警 loop 冲突),
    # 按仓库惯例改 httpx ASGITransport 同 loop 直跑;db_clean 挂夹具上兜跨用例残留。
    app = FastAPI()
    app.include_router(review_api.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def stub_kb(monkeypatch):
    written = {}

    async def fake_write(chunks):
        written["chunks"] = chunks
        return [101]

    async def fake_vectorize():
        written["vectorized"] = True
        return 1

    monkeypatch.setattr(review_api, "_write_chunks", fake_write)
    monkeypatch.setattr(review_api, "_vectorize", fake_vectorize)
    return written


async def _seed():
    rid = await repository.insert_review_item("猫窝是否支持水洗", "可以,以平台售后规则为准")
    lcid = await repository.insert_low_confidence(
        None, "猫窝可以洗吗", "user_feedback", None,
        retrieved_chunks=[{"question": "q", "answer": "a", "rerank_score": 0.3, "section_path": "s"}])
    await repository.set_matched_review(lcid, rid)
    return rid


async def test_queue_lists_pending(client):
    rid = await _seed()
    r = await client.get("/api/review/queue", params={"status": "待审"})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items and items[0]["id"] == rid and items[0]["occurrence_count"] == 1


async def test_detail_carries_raws_and_snapshot(client):
    rid = await _seed()
    r = await client.get(f"/api/review/{rid}")
    assert r.status_code == 200
    d = r.json()
    assert d["normalized_question"] == "猫窝是否支持水洗"
    assert d["raws"][0]["source"] == "user_feedback"
    assert d["raws"][0]["retrieved_chunks"][0]["rerank_score"] == 0.3
    assert (await client.get("/api/review/99999")).status_code == 404


async def test_approve_writes_kb_and_flips_status(client, stub_kb):
    rid = await _seed()
    r = await client.post(f"/api/review/{rid}/approve", json={"approved_answer": "可以水洗,注意阴干"})
    assert r.status_code == 200 and r.json()["chunk_ids"] == [101]
    chunk = stub_kb["chunks"][0]
    assert chunk.questions == "猫窝是否支持水洗" and chunk.answer == "可以水洗,注意阴干"
    assert chunk.content_type == "faq"
    assert stub_kb["vectorized"] is True
    r2 = await client.get(f"/api/review/{rid}")
    assert r2.json()["review_status"] == "通过"
    # 终审后再操作 → 409
    assert (await client.post(f"/api/review/{rid}/reject")).status_code == 409


async def test_approve_failure_keeps_pending(client, monkeypatch):
    rid = await _seed()

    async def boom(chunks):
        raise RuntimeError("嵌入上游不可用")

    monkeypatch.setattr(review_api, "_write_chunks", boom)
    r = await client.post(f"/api/review/{rid}/approve", json={"approved_answer": "x"})
    assert r.status_code == 502
    r2 = await client.get(f"/api/review/{rid}")
    assert r2.json()["review_status"] == "待审"   # 可重试


async def test_approve_retry_clears_orphan_pending_chunks(client, monkeypatch):
    """写回失败留下的 pending 残件,重试通过时必须被清掉——vectorize_pending 捞全部 pending,
    不清会把残件和新件一起送进 Milvus,知识库从此检索出重复证据(ch09 review #2)。"""
    rid = await _seed()
    section_path = "飞轮沉淀 / 猫窝是否支持水洗"

    async def real_write(chunks):   # 第一腿不 stub:真插一行 pending,复现残件现场
        c = chunks[0]
        cid = await repository.insert_knowledge_chunk(
            c.category, c.questions, c.answer,
            section_path=c.section_path, content_type=c.content_type, is_key_clause=c.is_key_clause)
        return [cid]

    async def boom():
        raise RuntimeError("Milvus 不可用")

    monkeypatch.setattr(review_api, "_write_chunks", real_write)
    monkeypatch.setattr(review_api, "_vectorize", boom)
    assert (await client.post(f"/api/review/{rid}/approve",
                              json={"approved_answer": "可以水洗"})).status_code == 502
    pending = [c for c in await repository.list_pending_chunks() if c.section_path == section_path]
    assert len(pending) == 1                     # 失败现场:残件在库

    async def fake_write(chunks):   # 第二腿 stub:新件不落真库,断言聚焦「残件被清」
        return [202]

    async def fake_vectorize():
        return 1

    monkeypatch.setattr(review_api, "_write_chunks", fake_write)
    monkeypatch.setattr(review_api, "_vectorize", fake_vectorize)
    assert (await client.post(f"/api/review/{rid}/approve",
                              json={"approved_answer": "可以水洗"})).status_code == 200
    assert not [c for c in await repository.list_pending_chunks() if c.section_path == section_path]


async def test_claim_review_is_one_shot(db_session_factory, db_clean):
    rid = await repository.insert_review_item("q", "a")
    assert await repository.claim_review(rid, "第一次") is True
    assert await repository.claim_review(rid, "第二次") is False   # 并发双击只有一个过闸
    row = await repository.get_review_detail(rid)
    assert row[0].approved_answer == "第一次"                     # 后到的写不进去


async def test_reject_flips_status(client):
    rid = await _seed()
    assert (await client.post(f"/api/review/{rid}/reject")).status_code == 200
    r = await client.get(f"/api/review/{rid}")
    assert r.json()["review_status"] == "驳回"
    assert (await client.post(f"/api/review/{rid}/approve", json={"approved_answer": "x"})).status_code == 409
