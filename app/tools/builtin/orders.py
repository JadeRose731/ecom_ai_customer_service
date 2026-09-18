# app/tools/builtin/orders.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools import registry
from app.tools.business import order_snapshot


class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")


class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。"""
    return order_snapshot(order_id)


@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    import random
    rng = random.Random(f"product:{product_name}")
    return {"product_name": product_name, "price": rng.randint(20, 999),
            "stock": rng.randint(0, 500), "spec": rng.choice(["标准装", "家庭装", "试用装"])}


registry.register(registry.spec_from_langchain_tool(query_order, source="builtin"))
registry.register(registry.spec_from_langchain_tool(query_product, source="builtin"))
