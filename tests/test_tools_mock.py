# tests/test_tools_mock.py
from app.tools.builtin.orders import query_order, query_product

async def test_query_order_deterministic_and_shaped():
    r1 = await query_order.ainvoke({"order_id": "1001"})
    r2 = await query_order.ainvoke({"order_id": "1001"})
    assert r1 == r2                                  # 同种子可复现
    assert r1["order_id"] == "1001"
    assert r1["status"] in {"待付款", "已付款", "已发货", "已签收"}

async def test_query_product_shape():
    assert query_product.name == "query_product"
    p = await query_product.ainvoke({"product_name": "猫粮"})
    assert p["product_name"] == "猫粮" and "price" in p
    # ch08:内置 query_logistics 测试随工具下线删除(MCP 侧见 tests/tools/test_mcp_servers.py)
