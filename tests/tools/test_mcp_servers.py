"""ch08 MCP Server 集成测试:真起子进程 → adapters client 列工具 + 真调。
连不上视为环境问题直接 fail(两台 Server 是本章交付物,不 skip)。"""
import asyncio
import os
import subprocess
import sys
import time

import pytest_asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


def _client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "logistics": {"transport": "streamable_http", "url": "http://127.0.0.1:18101/mcp"},
        "aftersales": {"transport": "streamable_http", "url": "http://127.0.0.1:18102/mcp"},
    }, handle_tool_errors=False)


@pytest_asyncio.fixture(scope="module")
async def mcp_procs():
    env1 = {**os.environ, "PORT": "18101"}
    p1 = subprocess.Popen([sys.executable, "mcp_servers/logistics_server.py"], env=env1,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    env2 = {**os.environ, "PORT": "18102"}
    p2 = subprocess.Popen([sys.executable, "mcp_servers/aftersales_server.py"], env=env2,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 轮询探活(Plan 授权的 sleep 替代):get_tools 全 Server 连通才算就绪
    deadline = time.time() + 20
    last_err = None
    while time.time() < deadline:
        try:
            await _client().get_tools()
            break
        except Exception as e:  # noqa: BLE001 就绪前连不上属预期,重试
            last_err = e
            await asyncio.sleep(0.3)
    else:
        p1.terminate(); p2.terminate()
        raise RuntimeError(f"MCP Server 20s 未就绪: {type(last_err).__name__}: {last_err}")
    yield
    p1.terminate(); p2.terminate()
    p1.wait(timeout=5); p2.wait(timeout=5)


async def test_list_tools_three_essentials(mcp_procs):
    tools = await _client().get_tools()
    by = {t.name: t for t in tools}
    assert set(by) == {"query_logistics", "query_warranty", "query_return_status"}
    for t in by.values():
        assert t.description                                   # 用途描述
        schema = t.args_schema if isinstance(t.args_schema, dict) else t.args_schema.model_json_schema()
        assert schema.get("properties")                        # JSON Schema 参数定义


async def test_invoke_logistics_stable_mock(mcp_procs):
    tools = {t.name: t for t in await _client().get_tools(server_name="logistics")}
    r1 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    r2 = await tools["query_logistics"].ainvoke({"tracking_no": "SF123"})
    # adapters 0.3.2 实装(dev-notes Task 4 裁定):ainvoke 返回 content blocks 列表,
    # 每块带本次调用唯一的 lc_ id,列表级相等必假——稳定性断言落在业务 JSON 文本上
    assert isinstance(r1, list) and r1 and r1[0]["type"] == "text"
    assert r1[0]["text"] == r2[0]["text"]                      # 种子稳定
    assert "status_code" in r1[0]["text"]                      # 内部枚举码在
