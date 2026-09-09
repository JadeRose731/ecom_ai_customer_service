# tests/test_faith_cases_api.py
"""编造个案台账:一题一行 / 复发退回 / 分页筛选 / 证据快照 / 处置必填 / 退回清说明 / 非法状态 422,
外加幻觉率口径——台账累计 3 条、本轮只判出 1 条时,判出率是 1/300(不是 3/300)。"""
import pytest

from app.db import repository
from app.main import app
from scripts.eval_ch04 import hallucination_round


def _citations() -> list[dict]:
    return [
        {"n": 1, "id": 12, "question": "退货政策是什么",
         "answer": "支持 7 天无理由退货", "section_path": "商品与购物 FAQ / 退货政策是什么"},
        {"n": 2, "id": 13, "question": "退款时效",
         "answer": "退款 3 个工作日内原路退回", "section_path": "商品与购物 FAQ / 退款时效"},
    ]


async def _get(client, url):
    return await client.get(url)


async def _post(client, url, json):
    return await client.post(url, json=json)


@pytest.mark.asyncio
async def test_one_row_per_question(client, db_session_factory, db_clean):
    """同一题判出两轮:仍是一行,seen_count=2,正文换成最新一轮。"""
    r1 = await repository.upsert_faith_case("C1", "旧问法", "旧答案", _citations())
    r2 = await repository.upsert_faith_case("C1", "新问法", "新答案", None)
    assert r2["id"] == r1["id"] and not r2["relapsed"]
    data = (await _get(client, "/api/rag-eval/faith-cases")).json()
    assert data["total"] == 1
    item = data["items"][0]
    assert item["seen_count"] == 2 and item["query"] == "新问法" and item["answer"] == "新答案"


@pytest.mark.asyncio
async def test_relapse_reopens_resolved_case(client, db_session_factory, db_clean):
    """已处置的个案再判出来:退回未解决、清说明与处置时间、带复发标记。"""
    created = await repository.upsert_faith_case("E15", "q", "a", None)
    await _post(client, f"/api/rag-eval/faith-cases/{created['id']}/status",
                {"status": "resolved", "resolution": "已补语料"})
    again = await repository.upsert_faith_case("E15", "q2", "a2", None)
    assert again["relapsed"] is True
    item = (await _get(client, "/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "unresolved" and item["resolution"] is None
    assert item["resolved_at"] is None and item["seen_count"] == 2


@pytest.mark.asyncio
async def test_pagination_and_status_filter(client, db_session_factory, db_clean):
    for eid in ("A1", "A2", "A3"):
        await repository.upsert_faith_case(eid, "q", "a", None)
    first = await repository.upsert_faith_case("B1", "q", "a", None)
    await _post(client, f"/api/rag-eval/faith-cases/{first['id']}/status",
                {"status": "wontfix", "resolution": "口径如此"})
    page = (await _get(client, "/api/rag-eval/faith-cases?page=1&size=2")).json()
    assert page["total"] == 4 and len(page["items"]) == 2
    assert page["counts"] == {"unresolved": 3, "resolved": 0, "wontfix": 1}
    # 未解决排前
    assert all(i["status"] == "unresolved" for i in page["items"])
    page2 = (await _get(client, "/api/rag-eval/faith-cases?page=2&size=2")).json()
    assert [i["status"] for i in page2["items"]] == ["unresolved", "wontfix"]
    only_open = (await _get(client, "/api/rag-eval/faith-cases?status=unresolved")).json()
    assert only_open["total"] == 3


@pytest.mark.asyncio
async def test_citations_snapshot_roundtrip(client, db_session_factory, db_clean):
    """证据快照如实带出:顺序、编号、chunk id、section_path 原样。"""
    await repository.upsert_faith_case("B7", "q", "a", _citations())
    item = (await _get(client, "/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["citations"] == _citations()


@pytest.mark.asyncio
async def test_resolution_required(client, db_session_factory, db_clean):
    created = await repository.upsert_faith_case("A5", "q", "a", None)
    url = f"/api/rag-eval/faith-cases/{created['id']}/status"
    assert (await _post(client, url, {"status": "resolved", "resolution": "   "})).status_code == 400
    assert (await _post(client, url, {"status": "resolved"})).status_code == 400
    assert (await _post(client, url, {"status": "wontfix", "resolution": "裁判口径豁免"})).status_code == 200
    item = (await _get(client, "/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "wontfix" and item["resolution"] == "裁判口径豁免"
    assert item["resolved_at"] is not None


@pytest.mark.asyncio
async def test_reopen_clears_resolution(client, db_session_factory, db_clean):
    created = await repository.upsert_faith_case("A6", "q", "a", None)
    url = f"/api/rag-eval/faith-cases/{created['id']}/status"
    await _post(client, url, {"status": "resolved", "resolution": "改了语料"})
    assert (await _post(client, url, {"status": "unresolved"})).status_code == 200
    item = (await _get(client, "/api/rag-eval/faith-cases")).json()["items"][0]
    assert item["status"] == "unresolved" and item["resolution"] is None
    assert item["resolved_at"] is None


@pytest.mark.asyncio
async def test_illegal_status_rejected(client, db_session_factory, db_clean):
    """Literal 挡脏值:pydantic 校验直接 422,不打到库。"""
    created = await repository.upsert_faith_case("A7", "q", "a", None)
    resp = await _post(client, f"/api/rag-eval/faith-cases/{created['id']}/status",
                       {"status": "closed", "resolution": "x"})
    assert resp.status_code == 422
    assert app.routes  # 路由仍存活(未被脏状态打断)


@pytest.mark.asyncio
async def test_unknown_case_404(client, db_session_factory, db_clean):
    resp = await _post(client, "/api/rag-eval/faith-cases/9999/status",
                       {"status": "resolved", "resolution": "x"})
    assert resp.status_code == 404


def test_hallucination_rate_counts_current_round_only():
    """台账累计 3 条、本轮只判出 1 条:判出率 = 1/300,台账 3 条单独进 ledger。"""
    status_map = {"C2": "unresolved", "E15": "resolved", "A9": "wontfix"}
    block = hallucination_round(["C2"], 300, status_map)
    assert block["round"]["judged"] == 1
    assert block["round"]["rate"] == pytest.approx(1 / 300, abs=1e-4)
    assert block["ledger"]["total"] == 3
    assert block["ledger"]["unresolved"] == 1
