# app/tools/builtin/refunds.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry


class RefundInput(BaseModel):
    order_id: str = Field(description="要退款的订单号")
    reason: str | None = Field(default=None, description="退款原因(可选,最终以前端固定类目下拉为准)")


@tool(args_schema=RefundInput)
async def submit_refund(order_id: str, reason: str | None = None) -> dict:
    """判定这一单可以退款后,调用本工具发起退款申请。实际提交由前端退款表单确认后落库,
    本工具只表示『这一单可以退,已把提交入口交给用户』。"""
    return {"status": "待用户确认", "order_id": order_id}


registry.register(registry.spec_from_langchain_tool(submit_refund, source="builtin"))
