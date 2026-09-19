"""ch09 审核后台 API:待审队列列表/详情/通过(写回知识库)/驳回。
通过 = 原子占位状态(条件 UPDATE,并发双击只有一个过闸)→ 清 pending 残件 → 核准答案以
QA chunk 走 ch03 落库流程(write_pending → vectorize_pending 同步);写回失败退回待审可重试,
重试前清残件保证 Milvus 不出重复证据——审核页点了通过,下一问就能检索命中(验收 3)。"""
import logging

from fastapi import APIRouter, HTTPException

from app.db import repository
from app.kb import dualwrite
from app.kb.documents import Chunk, _is_key
from app.schemas.review import ApproveRequest

logger = logging.getLogger(__name__)
router = APIRouter()


async def _write_chunks(chunks: list[Chunk]) -> list[int]:
    return await dualwrite.write_pending(chunks)


async def _vectorize() -> int:
    return await dualwrite.vectorize_pending()


def _item_out(r) -> dict:
    return {"id": r.id, "normalized_question": r.normalized_question,
            "ai_suggested_answer": r.ai_suggested_answer,
            "occurrence_count": r.occurrence_count, "review_status": r.review_status,
            "created_at": r.created_at.isoformat() if r.created_at else None}


@router.get("/api/review/queue")
async def review_queue(status: str | None = None) -> dict:
    rows = await repository.list_review_queue(status)
    return {"items": [_item_out(r) for r in rows]}


@router.get("/api/review/{review_id}")
async def review_detail(review_id: int) -> dict:
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, raws = detail
    out = _item_out(item)
    out["approved_answer"] = item.approved_answer
    out["raws"] = [{"raw_question": r.raw_question, "source": r.source, "reason": r.reason,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "retrieved_chunks": r.retrieved_chunks}
                   for r in raws]
    return out


@router.post("/api/review/{review_id}/approve")
async def approve(review_id: int, req: ApproveRequest) -> dict:
    detail = await repository.get_review_detail(review_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="缺口不存在")
    item, _ = detail
    if item.review_status != "待审":
        raise HTTPException(status_code=409, detail=f"当前状态为「{item.review_status}」,不可再审")

    # 先原子占位(条件 UPDATE,并发双击只有一个能过闸),过了闸才允许写知识库
    if not await repository.claim_review(review_id, req.approved_answer):
        raise HTTPException(status_code=409, detail="仅待审状态可通过")

    section_path = f"飞轮沉淀 / {item.normalized_question}"
    chunk = Chunk(
        category="飞轮沉淀", questions=item.normalized_question, answer=req.approved_answer,
        section_path=section_path, content_type="faq",
        is_key_clause=_is_key(item.normalized_question, req.approved_answer),
    )
    try:
        # 清上次写回失败留下的 pending 残件——不清,vectorize_pending 会把残件和新件都送进 Milvus
        await repository.delete_pending_chunks_by_section(section_path)
        chunk_ids = await _write_chunks([chunk])
        await _vectorize()
    except Exception:
        logger.exception("审核写回知识库失败 review=%s(状态退回待审,可重试)", review_id)
        await repository.revert_review_claim(review_id)
        raise HTTPException(status_code=502, detail="写回知识库失败(检查上游/Milvus),状态未变可重试")

    logger.info("审核通过 review=%s → knowledge_chunks %s(已向量化,下一问可检索)", review_id, chunk_ids)
    return {"ok": True, "chunk_ids": chunk_ids}


@router.post("/api/review/{review_id}/reject")
async def reject(review_id: int) -> dict:
    if not await repository.update_review_status(review_id, "驳回"):
        detail = await repository.get_review_detail(review_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="缺口不存在")
        raise HTTPException(status_code=409, detail="仅待审状态可驳回")
    return {"ok": True}
