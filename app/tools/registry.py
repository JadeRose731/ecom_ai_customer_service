# app/tools/registry.py
"""ch08 工具注册中心:内置(启动扫描 builtin/ 包)+ MCP(mcp_client 现拉)统一登记 ToolSpec。
三样必齐:工具名、用途描述、JSON Schema 参数定义。权限只认我们侧 WRITE_TOOLS,不看 Server 声明。"""
import importlib
import logging
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 权限我们侧独裁:写清单按名字定;未登记(含所有 MCP)工具一律按只读放行。
# 生产环境接不受信 Server 时应默认拒绝未知写操作;本章两台自建 Server 都是查询类,只读放行足够。
WRITE_TOOLS: set[str] = {"create_ticket"}


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict            # 参数 JSON Schema(校验用;模型可见口径,不含注入参数)
    tool: BaseTool
    permission: str              # "read" | "write"
    source: str                  # "builtin" | "mcp"
    mcp_server: str | None = None
    timeout: float | None = None            # None → engine 按来源取默认
    inject_conversation: bool = False       # 执行前注入 conversation_id
    format_result: Callable[[dict], dict] | None = None  # 挑字段/枚举翻人话,None 透传


_BUILTIN: dict[str, ToolSpec] = {}
_scanned = False


def permission_for(name: str) -> str:
    return "write" if name in WRITE_TOOLS else "read"


def _json_schema_of(tool: BaseTool) -> dict:
    """三样齐的 schema 口径:MCP 工具 args_schema 本就是 dict;内置 pydantic 模型用
    tool_call_schema(排除 InjectedToolArg,模型可见口径)转 JSON Schema。"""
    raw = getattr(tool, "args_schema", None)
    if isinstance(raw, dict):
        return raw
    tcs = getattr(tool, "tool_call_schema", None) or raw
    return tcs.model_json_schema()


def spec_from_langchain_tool(tool: BaseTool, *, source: str, mcp_server: str | None = None,
                             timeout: float | None = None, inject_conversation: bool = False,
                             format_result: Callable | None = None) -> ToolSpec:
    return ToolSpec(name=tool.name, description=tool.description or "",
                    json_schema=_json_schema_of(tool), tool=tool,
                    permission=permission_for(tool.name), source=source, mcp_server=mcp_server,
                    timeout=timeout, inject_conversation=inject_conversation,
                    format_result=format_result)


def register(spec: ToolSpec) -> None:
    if spec.name in _BUILTIN:
        logger.warning("工具重名,丢弃后注册者 name=%s(先到者保留)", spec.name)
        return
    _BUILTIN[spec.name] = spec


def scan_builtin() -> None:
    """服务启动(lifespan)时调用:导入 builtin/ 包全部模块,模块 import 即注册。幂等。"""
    global _scanned
    if _scanned:
        return
    _scanned = True
    from app.tools import builtin as pkg
    for m in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{m.name}")
    logger.info("内置工具注册完成:%s", sorted(_BUILTIN))


def builtin_specs() -> list[ToolSpec]:
    scan_builtin()
    return list(_BUILTIN.values())


def get_builtin_spec(name: str) -> ToolSpec | None:
    scan_builtin()
    return _BUILTIN.get(name)


async def get_all_specs() -> list[ToolSpec]:
    """内置 + MCP 现拉合并;重名 builtin 优先、后到丢弃告警(本章无重名:内置 query_logistics 已下线)。"""
    from app.tools import mcp_client   # 延迟导入避免环
    merged: dict[str, ToolSpec] = {s.name: s for s in builtin_specs()}
    for s in await mcp_client.fetch_mcp_specs():
        if s.name in merged:
            logger.warning("MCP 工具与已注册工具重名,丢弃 name=%s server=%s", s.name, s.mcp_server)
            continue
        merged[s.name] = s
    return list(merged.values())
