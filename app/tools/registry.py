# app/tools/registry.py
from langchain_core.tools import BaseTool

from app.tools.business import (
    create_ticket,
    query_faq,
    query_logistics,
    query_order,
    query_product,
    submit_refund,
)

_ALL: list[BaseTool] = [query_order, query_product, query_logistics, query_faq, create_ticket, submit_refund]
_BY_NAME: dict[str, BaseTool] = {t.name: t for t in _ALL}

NO_RETRY: set[str] = {"create_ticket", "submit_refund"}  # 写类工具不自动重试(被拦截、从不真执行,列入无妨)
INJECT_CONVERSATION: set[str] = {"create_ticket"}  # 需注入会话主键的工具

def get_all_tools() -> list[BaseTool]:
    return _ALL

def get_tool(name: str) -> BaseTool | None:
    return _BY_NAME.get(name)
