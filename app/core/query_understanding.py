# app/core/query_understanding.py
# ch04:检索前的 Query 理解——口语归一 + 同义词扩展。只作用于检索侧,不改入库。
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开 glm-5.2 502。"""
    model = get_chat_model().with_structured_output(_Rewrite)
    r: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}
