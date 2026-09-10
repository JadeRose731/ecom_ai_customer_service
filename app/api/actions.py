# app/api/actions.py
import logging

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.db import repository
from app.schemas.actions import CreateTicketRequest, CreateTicketResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/actions/create-ticket", response_model=CreateTicketResponse)
async def create_ticket_action(req: CreateTicketRequest) -> CreateTicketResponse:
    """建工单按钮:用户点了才写 tickets 表(复用 ch02 工单能力)。转人工是另一件事,纯前端。"""
    try:
        ticket_no = await repository.create_ticket(
            req.conversation_id, req.description, req.ticket_type)
    except SQLAlchemyError:
        logger.exception("建工单失败 conv=%s", req.conversation_id)
        raise HTTPException(status_code=503, detail="工单系统暂时不可用,请稍后重试")
    return CreateTicketResponse(ticket_no=ticket_no)
