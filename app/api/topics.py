"""ch10 主题分布 API:飞轮后台看各类目问题量,决定先补哪块知识。只读。"""
from fastapi import APIRouter, HTTPException

from app.core.taxonomy import LABEL2ID
from app.db import repository

router = APIRouter()


@router.get("/api/topics/distribution")
async def distribution():
    return await repository.topic_distribution()


@router.get("/api/topics/questions")
async def questions(label: str, page: int = 1, size: int = 20):
    if label not in LABEL2ID:
        raise HTTPException(status_code=400, detail=f"未知类目:{label}")
    return await repository.topic_questions(label, page, size)
