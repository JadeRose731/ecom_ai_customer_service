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
    # 4+ 位纯数字,与 mock 订单域(list_user_orders 1000-9999)/ _extract_order_id 一致
    order_id: str = Field(min_length=1, max_length=32, pattern=r"^\d{4,}$")
    reason: Literal["七天无理由", "质量问题", "发错货", "不想要了", "其他"]


class CreateRefundResponse(BaseModel):
    ticket_no: str
    status: str = "退款申请已提交"


class ResumeRequest(BaseModel):
    conversation_id: int
    order_id: str = Field(min_length=1, max_length=32, pattern=r"^\d{4,}$")
