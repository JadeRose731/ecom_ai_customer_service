"""ch10 主题分布 API:飞轮后台看各类目问题量,决定先补哪块知识。只读。"""
from fastapi import APIRouter, HTTPException, Query

from app.core.taxonomy import LABEL2ID
from app.db import repository

router = APIRouter()


@router.get("/api/topics/distribution")
async def distribution():
    return await repository.topic_distribution()


@router.get("/api/topics/questions")
async def questions(label: str, page: int = Query(1, gt=0), size: int = Query(20, gt=0, le=100)):
    if label not in LABEL2ID:
        raise HTTPException(status_code=400, detail=f"未知类目:{label}")
    return await repository.topic_questions(label, page, size)
