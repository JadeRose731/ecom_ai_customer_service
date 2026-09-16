# app/schemas/actions.py
from typing import Literal

from pydantic import BaseModel, Field


class CreateTicketRequest(BaseModel):
    conversation_id: int
    description: str = Field(min_length=1)
    ticket_type: Literal["售后", "投诉", "咨询"]


class CreateTicketResponse(BaseModel):
    ticket_no: str
    status: str = "已转人工"


class CreateRefundRequest(BaseModel):
    conversation_id: int
    order_id: str = Field(min_length=1)
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]


class CreateRefundResponse(BaseModel):
    ticket_no: str
    status: str = "退款申请已提交"
