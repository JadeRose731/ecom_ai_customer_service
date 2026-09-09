# app/core/selfcheck.py
# ch04:生成前的证据充分性自评闸——先判够不够,不够就不进生成。
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import SELF_CHECK_PROMPT


class _Check(BaseModel):
    useful: bool = Field(description="证据是否足以回答")
    reason: str = Field(default="", description="判断依据")


def _chain():
    return SELF_CHECK_PROMPT | get_chat_model().with_structured_output(_Check)


async def check_sufficient(query: str, evidence_texts: list[str]) -> dict:
    """生成前证据充分性自评。扁平字段 useful/reason(避开 glm-5.2 嵌套数组 502)。"""
    evidence = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(evidence_texts)) or "(无证据)"
    r: _Check = await _chain().ainvoke({"query": query, "evidence": evidence})
    return {"useful": bool(r.useful), "reason": r.reason or ""}
