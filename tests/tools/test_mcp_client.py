import pytest

from app.tools import mcp_client, registry
from app.tools.registry import ToolSpec


@pytest.fixture(autouse=True)
def _clean_dead_since():
    """判死冷却表是模块级状态,先清后测,避免用例间互相污染。"""
    mcp_client._dead_since.clear()
    yield
    mcp_client._dead_since.clear()


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = f"{name} 描述"
        self.args_schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


async def test_fetch_mcp_specs_marks_source_and_permission(monkeypatch):
    async def fake_get_tools(*, server_name=None):
        return [_FakeTool("query_logistics")] if server_name == "logistics" else [_FakeTool("query_warranty")]
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    by = {s.name: s for s in specs}
    assert by["query_logistics"].source == "mcp" and by["query_logistics"].mcp_server == "logistics"
    assert all(s.permission == "read" for s in specs)          # 我们侧规则:MCP 不在写清单→只读
    assert by["query_logistics"].format_result is mcp_client.FORMATTERS["query_logistics"]


async def test_one_server_down_degrades_gracefully(monkeypatch, caplog):
    async def fake_get_tools(*, server_name=None):
        if server_name == "logistics":
            raise ConnectionError("拒绝连接")
        return [_FakeTool("query_warranty")]
    monkeypatch.setattr(mcp_client, "_get_tools_of", fake_get_tools)
    specs = await mcp_client.fetch_mcp_specs()
    assert {s.name for s in specs} == {"query_warranty"}       # 单台挂了跳过,不拖垮
    assert any("不可达" in r.message for r in caplog.records)


async def test_dead_cooldown_skips_then_reprobes(monkeypatch):
    """判死冷却(ch08 评审 Important#2):失败判死 → 冷却期内不再发起连接 → 到期重探,
    成功即清除判死。回归此处会让恢复了的 Server 静默失踪整整一个 30s 窗口。"""
    calls = []

    async def flaky(*, server_name=None):
        calls.append(server_name)
        if server_name == "logistics" and len(calls) == 1:
            raise ConnectionError("拒绝连接")
        return []
    monkeypatch.setattr(mcp_client, "_get_tools_of", flaky)

    await mcp_client.fetch_mcp_specs()
    assert mcp_client._dead_since.get("logistics") is not None      # 首败判死

    await mcp_client.fetch_mcp_specs()
    assert calls.count("logistics") == 1                            # 冷却期内物流零连接

    # 到期重探:把判死时刻拨回窗口之前(不动时钟,patch 全局 time.monotonic 会波及 asyncio 内部)
    mcp_client._dead_since["logistics"] -= mcp_client._COOLDOWN_SECONDS + 1
    specs = await mcp_client.fetch_mcp_specs()
    assert calls.count("logistics") == 2
    assert "logistics" not in mcp_client._dead_since                # 成功即清除
    assert isinstance(specs, list)


async def test_get_all_specs_merges_builtin_wins(monkeypatch):
    async def fake_fetch():
        return [ToolSpec(name="query_order", description="MCP 冒名", json_schema={"type": "object", "properties": {}},
                         tool=_FakeTool("query_order"), permission="read", source="mcp", mcp_server="x"),
                ToolSpec(name="query_logistics", description="物流", json_schema={"type": "object", "properties": {}},
                         tool=_FakeTool("query_logistics"), permission="read", source="mcp", mcp_server="logistics")]
    monkeypatch.setattr(mcp_client, "fetch_mcp_specs", fake_fetch)
    specs = await registry.get_all_specs()
    by = {s.name: s for s in specs}
    assert by["query_order"].source == "builtin"               # 重名 builtin 优先,后到 MCP 丢弃
    assert by["query_logistics"].source == "mcp"               # 物流已下线内置,由 MCP 接管
    assert {"query_faq", "create_ticket", "submit_refund", "query_product"} <= set(by)


def test_logistics_formatter_translates_codes():
    out = mcp_client.FORMATTERS["query_logistics"](
        {"tracking_no": "SF1", "status_code": "IN_TRANSIT", "current_city": "深圳",
         "trace": ["深圳分拨中心 已发出"], "carrier_code": "SF-EXP-01"})
    assert out == {"tracking_no": "SF1", "status": "运输中", "current_city": "深圳",
                   "trace": ["深圳分拨中心 已发出"]}            # 挑字段 + 枚举翻人话,内部编码剔除
