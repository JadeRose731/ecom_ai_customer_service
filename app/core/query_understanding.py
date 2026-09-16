# app/core/query_understanding.py
# ch04:检索前的 Query 理解——口语归一 + 同义词扩展。只作用于检索侧,不改入库。
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import EXPAND_QUERIES_PROMPT, QUERY_REWRITE_PROMPT


class _Rewrite(BaseModel):
    standard: str = Field(description="标准问法")
    expanded: list[str] = Field(default_factory=list, description="同义/近义扩展词")


async def understand(query: str) -> dict:
    """口语→标准问法 + 同义词扩展。扁平字段(list[str],非嵌套对象),避开 glm-5.2 502。"""
    model = get_chat_model().with_structured_output(_Rewrite)
    r: _Rewrite = await (QUERY_REWRITE_PROMPT | model).ainvoke({"query": query})
    return {"standard": r.standard or query, "expanded": list(r.expanded or [])}


class _Expanded(BaseModel):
    queries: list[str] = Field(default_factory=list, description="严格3条检索友好查询")


async def expand_queries(query: str) -> list[str]:
    """把一句问题泛化成检索友好查询(强制 JSON 单字段,扁平 list[str] 避开 glm 502)。
    严格取前 3 条非空;模型异常/全空则回落 [query],保证 retrieve_policy 至少有一条可查。"""
    model = get_chat_model().with_structured_output(_Expanded)
    try:
        r: _Expanded = await (EXPAND_QUERIES_PROMPT | model).ainvoke({"query": query})
        qs = [q.strip() for q in (r.queries or []) if q and q.strip()]
    except Exception:
        qs = []
    return qs[:3] or [query]
