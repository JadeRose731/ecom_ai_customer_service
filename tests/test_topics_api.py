"""ch10 主题 API:分布(17 类全出/计数/样例去重)+ 类目问题列表(分页/多标签/归并状态/400)。"""
import httpx
import pytest
from fastapi import FastAPI

from app.api import topics as topics_api
from app.db import repository


@pytest.fixture
async def client(db_session_factory, db_clean):
    # conftest 全局 client 不挂 db_clean,主题测试要种池/归类数据,照 review 惯例本地起
    app = FastAPI()
    app.include_router(topics_api.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def _seed_classified(text: str, labels: list[str], *, review: str | None = None) -> int:
    """种一条池问题(可选归并到审核行)并写归类,返回 question_id。"""
    rid = await repository.insert_review_item(review, "答案") if review else None
    lid = await repository.insert_low_confidence(None, text, "retrieval_low_conf", None)
    if rid is not None:
        await repository.set_matched_review(lid, rid)
    await repository.insert_topic_classifications([{"question_id": lid, "labels": labels}])
    return lid


async def test_distribution_empty_returns_all_17(client):
    resp = await client.get("/api/topics/distribution")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert len(body["classes"]) == 17
    assert body["latest"] is None


async def test_distribution_counts(client):
    await _seed_classified("猫窝买大了想退", ["尺码", "退换货"])
    resp = await client.get("/api/topics/distribution")
    body = resp.json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert body["total"] == 1
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["物流"]["count"] == 0


async def test_questions_pagination_counts_per_class(client):
    await _seed_classified("甲一号", ["尺码"])
    await _seed_classified("甲二号", ["尺码"])
    await _seed_classified("甲三号", ["尺码", "物流"])
    await _seed_classified("乙一号", ["物流"])
    r = await client.get("/api/topics/questions", params={"label": "尺码", "page": 1, "size": 2})
    body = r.json()
    assert (body["label"], body["total"], body["page"], body["pages"], body["size"]) == \
        ("尺码", 3, 1, 2, 2)
    assert len(body["items"]) == 2
    r2 = await client.get("/api/topics/questions", params={"label": "尺码", "page": 2, "size": 2})
    assert r2.json()["total"] == 3 and len(r2.json()["items"]) == 1


async def test_questions_multi_label_appears_in_each_class(client):
    await _seed_classified("猫窝买大了想退", ["尺码", "退换货"])
    for lb, companion in (("尺码", "退换货"), ("退换货", "尺码")):
        r = await client.get("/api/topics/questions", params={"label": lb})
        items = r.json()["items"]
        assert len(items) == 1
        assert lb in items[0]["labels"] and companion in items[0]["labels"]


async def test_questions_unmerged_shows_raw(client):
    await _seed_classified("猫窝买大了想退", ["尺码"])
    r = await client.get("/api/topics/questions", params={"label": "尺码"})
    item = r.json()["items"][0]
    assert item["text"] == "猫窝买大了想退"
    assert item["normalized"] is False
    assert item["review_status"] is None


async def test_questions_merged_shows_normalized_and_status(client):
    await _seed_classified("猫窝可以洗吗", ["其他"], review="猫窝是否支持水洗")
    r = await client.get("/api/topics/questions", params={"label": "其他"})
    item = r.json()["items"][0]
    assert item["text"] == "猫窝是否支持水洗"
    assert item["raw"] == "猫窝可以洗吗"
    assert item["normalized"] is True
    assert item["review_status"] == "待审"


async def test_questions_unknown_label_returns_400(client):
    r = await client.get("/api/topics/questions", params={"label": "不存在的类目"})
    assert r.status_code == 400


async def test_questions_item_carries_source_occurrence_classified_at(client):
    # 列表页要展示:来源 / 同义合并条数 / 归类时间
    await _seed_classified("猫窝买大了想退", ["尺码"], review="猫窝尺寸问题")
    item = (await client.get("/api/topics/questions", params={"label": "尺码"})).json()["items"][0]
    assert item["source"] == "retrieval_low_conf"
    assert item["occurrence_count"] == 1
    assert item["classified_at"]  # ISO 时间串存在


async def test_distribution_samples_dedupe_by_text(client):
    # 同一标准化问法对应池里两行(两次低置信),都归「尺码」:样例去重,计数不减
    for raw in ("猫窝买大了想退一号", "猫窝买大了想退二号"):
        lid = await repository.insert_low_confidence(None, raw, "retrieval_low_conf", None)
        rid = await repository.insert_review_item("猫窝买大了想退", "答案")
        await repository.set_matched_review(lid, rid)
        await repository.insert_topic_classifications(
            [{"question_id": lid, "labels": ["尺码"]}])
    body = (await client.get("/api/topics/distribution")).json()
    by_label = {c["label"]: c for c in body["classes"]}
    assert by_label["尺码"]["count"] == 2
    assert by_label["尺码"]["samples"] == ["猫窝买大了想退"]
