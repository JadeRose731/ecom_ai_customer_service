from typing import Literal

from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import INTENT_CLASSIFY_PROMPT

INTENTS = ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊")


class _Intent(BaseModel):
    intent: Literal["物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"] = Field(
        description="七类意图之一"
    )


async def classify(query: str) -> str:
    """单标签七类意图。扁平字段避开 glm 嵌套 502;越界/异常回退闲聊(最保守出口)。"""
    model = get_chat_model().with_structured_output(_Intent)
    try:
        r: _Intent = await (INTENT_CLASSIFY_PROMPT | model).ainvoke({"query": query})
    except Exception:
        return "闲聊"
    return r.intent if r.intent in INTENTS else "闲聊"
