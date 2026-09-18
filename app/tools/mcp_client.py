# app/tools/mcp_client.py
"""ch08 MCP Client:MultiServerMCPClient 多 Server 接入,现问现拿(每次现拉工具清单,
adapters 每次调用新建 session——Server 侧加工具,客服系统不重启即可见)。
权限/格式化只认我们侧:Server 自报的用途描述仅供模型参考,能不能调按 registry.WRITE_TOOLS。"""
import asyncio
import logging
import time

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.config import settings
from app.tools import registry
from app.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# 列工具是轻操作,但本机(Windows)实测单次 get_tools 稳定 ~3s——闸只防「真挂死」,
# 不能比正常握手紧,故取 tool 超时上限;Server 不在时 adapters 内部重试 ~6.6s/台,
# 由判死冷却兜住重复开销。两者叠加:两台全挂最坏首轮 ~13s,其后 30s 内零开销。
_FETCH_DEADLINE = settings.mcp_tool_timeout
_COOLDOWN_SECONDS = 30.0   # 判死后的静默期:期内 fetch 直接跳过该 Server(不再发起连接),到期再探
_dead_since: dict[str, float] = {}   # server -> 判死时刻(成功即清除);「现问现拿」语义不变


def _httpx_factory(**kwargs) -> httpx.AsyncClient:
    """本机实测:trust_env=True 时单请求固定多 ~1.6s(httpx 读环境/系统代理与证书路径)。
    我们只连配置里的 localhost Server,不走系统代理才对——trust_env=False 关掉这层。"""
    kwargs.setdefault("trust_env", False)
    return httpx.AsyncClient(**kwargs)


def _translate(mapping: dict[str, str], code):
    return mapping.get(code, code)


def _fmt_logistics(d: dict) -> dict:
    return {"tracking_no": d.get("tracking_no"),
            "status": _translate({"PICKED_UP": "已揽件", "IN_TRANSIT": "运输中",
                                  "DELIVERING": "派送中", "DELIVERED": "已签收"}, d.get("status_code")),
            "current_city": d.get("current_city"), "trace": d.get("trace")}


def _fmt_warranty(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "warranty": _translate({"IN_WARRANTY": "在保", "EXPIRED": "已过保"}, d.get("warranty_code")),
            "warranty_until": d.get("warranty_until")}


def _fmt_return(d: dict) -> dict:
    return {"order_id": d.get("order_id"),
            "return_status": _translate({"AUDITING": "审核中", "RETURNING": "退货中",
                                         "REFUNDED": "已退款", "NONE": "无退货记录"}, d.get("return_code")),
            "updated_at": d.get("updated_at")}


# 结果格式化我们侧登记(挑回答用得上的字段 + 内部枚举码翻人话);未登记的 MCP 工具透传
FORMATTERS = {"query_logistics": _fmt_logistics, "query_warranty": _fmt_warranty,
              "query_return_status": _fmt_return}

_client: MultiServerMCPClient | None = None


def _connections() -> dict:
    return {
        "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url,
                      "httpx_client_factory": _httpx_factory},
        "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url,
                       "httpx_client_factory": _httpx_factory},
    }


def get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        # handle_tool_errors=False:工具错误抛 ToolException,由执行引擎统一分诊/回灌
        _client = MultiServerMCPClient(_connections(), handle_tool_errors=False)
    return _client


async def _get_tools_of(*, server_name: str):
    """薄壳:单测 monkeypatch 锚点。wait_for 闸:Server 不在时快失败,不吃 adapters 内部重试。"""
    return await asyncio.wait_for(get_client().get_tools(server_name=server_name),
                                  timeout=_FETCH_DEADLINE)


async def fetch_mcp_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for server in _connections():
        dead = _dead_since.get(server)
        if dead is not None and time.monotonic() - dead < _COOLDOWN_SECONDS:
            # 冷却期内直接跳过,不再发起连接(日志用 debug,别每轮刷屏);到期自然重探
            logger.debug("MCP Server「%s」冷却期内,本轮跳过(%.0fs 后重探)",
                         server, _COOLDOWN_SECONDS - (time.monotonic() - dead))
            continue
        try:
            tools = await _get_tools_of(server_name=server)
        except Exception as e:  # noqa: BLE001 单台不可达:判死进冷却,告警+跳过,不拖垮本轮对话
            _dead_since[server] = time.monotonic()
            logger.warning("MCP Server「%s」不可达,本轮跳过其工具(%.0fs 后重探):%s",
                           server, _COOLDOWN_SECONDS, type(e).__name__)
            continue
        _dead_since.pop(server, None)
        for t in tools:
            specs.append(registry.spec_from_langchain_tool(
                t, source="mcp", mcp_server=server, format_result=FORMATTERS.get(t.name)))
    return specs
