# app/api/conversations.py — ch07 会话侧栏:两个只读接口(列表 / 历史回载),前端任何失败静默降级
from fastapi import APIRouter, HTTPException, Query

from app.db import repository

router = APIRouter()


@router.get("/api/conversations")
async def list_conversations(user_id: str = Query(min_length=1)) -> dict:
    """某用户的会话列表(新在前,带首问预览 + 有无摘要标记),侧栏多会话切换用。"""
    items = await repository.list_conversations(user_id)
    return {"items": items}


@router.get("/api/conversations/{conversation_id}/messages")
async def list_conversation_messages(conversation_id: int) -> dict:
    """历史回载:user/assistant 逐条(role/content/created_at,tool 行不回前端);会话不存在 404。"""
    if await repository.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    msgs = await repository.list_dialog_messages(conversation_id)
    return {"items": [
        {"role": m.role, "content": m.content or "",
         "created_at": m.created_at.isoformat() if m.created_at else None}
        for m in msgs
    ]}
