"""ch10 地基:topic_classifications ORM 与池捞取/写表/分布聚合 repository。"""
from app.core.taxonomy import TOPIC_NAMES
from app.db import repository


async def _seed_merged(raw="猫窝买大了想退", norm="猫窝买大了能不能退货") -> int:
    """已归并的池问题:标准化问法挂 review_queue,原话挂 matched_review_id。"""
    rid = await repository.insert_review_item(norm, None)
    lcid = await repository.insert_low_confidence(None, raw, "retrieval_low_conf", None)
    await repository.set_matched_review(lcid, rid)
    return lcid


async def test_insert_and_exclude_classified(db_session_factory, db_clean):
    qid = await _seed_merged()  # 已归并才进分类
    rows = await repository.list_unclassified_questions()
    assert any(r["question_id"] == qid for r in rows)

    n = await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    assert n == 1
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)


async def test_unmerged_excluded_from_classify(db_session_factory, db_clean):
    qid = await repository.insert_low_confidence(
        None, "猫抓板回收吗", "retrieval_low_conf", None)  # 无归并 → 不进分类
    rows = await repository.list_unclassified_questions()
    assert not any(r["question_id"] == qid for r in rows)


async def test_text_prefers_normalized(db_session_factory, db_clean):
    qid = await _seed_merged(raw="这个还能退吗", norm="智能猫砂盆能否七天无理由退货")
    rows = await repository.list_pool_texts()
    row = next(r for r in rows if r["question_id"] == qid)
    assert row["text"] == "智能猫砂盆能否七天无理由退货"


async def test_topic_distribution_counts_all_17(db_session_factory, db_clean):
    qid = await _seed_merged()
    await repository.insert_topic_classifications(
        [{"question_id": qid, "labels": ["尺码", "退换货"]}])
    dist = await repository.topic_distribution()
    assert dist["total"] == 1
    assert [c["label"] for c in dist["classes"]] == list(TOPIC_NAMES)   # 17 类全出、顺序同权威表
    by_label = {c["label"]: c for c in dist["classes"]}
    assert by_label["尺码"]["count"] == 1 and by_label["退换货"]["count"] == 1
    assert by_label["尺码"]["samples"] == ["猫窝买大了能不能退货"]
