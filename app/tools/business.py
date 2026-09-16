# app/tools/business.py
import random
from typing import Annotated, Literal

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core import query_understanding, retrieval, selfcheck
from app.db import repository

class OrderInput(BaseModel):
    order_id: str = Field(description="订单号,例如 1001")

class ProductInput(BaseModel):
    product_name: str = Field(description="商品名称或关键词,例如 猫粮")

class LogisticsInput(BaseModel):
    order_id: str = Field(description="订单号,用于查询该订单的物流轨迹")

def order_snapshot(order_id: str) -> dict:
    """订单快照(纯函数,随机种子固定 → 同 order_id 稳定)。query_order 工具与 fetch_order 节点同源。"""
    rng = random.Random(f"order:{order_id}")
    return {
        "order_id": order_id,
        "status": rng.choice(["待付款", "已付款", "已发货", "已签收"]),
        "amount": rng.randint(50, 2000),
        "created_at": f"2026-07-{rng.randint(1, 12):02d} 10:00",
        "product": rng.choice(["智能猫砂盆", "猫粮 5kg", "猫爬架", "自动饮水机"]),
        "tracking_no": f"SF{rng.randint(10**11, 10**12 - 1)}",
    }


def list_user_orders(user_id: str) -> list[dict]:
    """按 user_id 稳定列出该用户的 2-4 笔订单(mock,不落库)。每笔用 order_snapshot 同源,
    前端选中后回填 order_id 即可 query_order。"""
    rng = random.Random(f"user_orders:{user_id}")
    ids = [str(rng.randint(1000, 9999)) for _ in range(rng.randint(2, 4))]
    out = []
    for oid in ids:
        s = order_snapshot(oid)
        out.append({"order_id": oid, "product": s["product"],
                    "status": s["status"], "amount": s["amount"],
                    "tracking_no": s["tracking_no"], "created_at": s["created_at"]})
    return out


@tool(args_schema=OrderInput)
async def query_order(order_id: str) -> dict:
    """查询订单的状态、金额、下单时间、商品名和物流单号(tracking_no)。用于用户询问某个订单情况时。
    要查物流轨迹,需先用本工具拿到订单的 tracking_no,再把它传给 query_logistics。"""
    return order_snapshot(order_id)

@tool(args_schema=ProductInput)
async def query_product(product_name: str) -> dict:
    """查询商品的价格、库存和规格。用于用户咨询某商品是否有货、多少钱时。"""
    rng = random.Random(f"product:{product_name}")
    return {
        "product_name": product_name,
        "price": rng.randint(20, 999),
        "stock": rng.randint(0, 500),
        "spec": rng.choice(["标准装", "家庭装", "试用装"]),
    }

@tool(args_schema=LogisticsInput)
async def query_logistics(order_id: str) -> dict:
    """查询订单的物流状态、当前位置和轨迹。用于用户询问物流/快递到哪了时。"""
    rng = random.Random(f"logistics:{order_id}")
    status = rng.choice(["已揽件", "运输中", "派送中", "已签收"])
    city = rng.choice(["深圳", "广州", "杭州", "上海", "成都"])
    return {
        "order_id": order_id,
        "status": status,
        "location": f"{city}分拨中心",
        "timeline": [f"{city}分拨中心 已发出", f"当前状态:{status}"],
    }

class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作类问题(可用原话)")
    category: str | None = Field(default=None, description="可选:按品类过滤,如『运费』『退货』『商品手册』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策知识库(混合检索+重排)。用于政策、规则、时效、费用、商品手册等通用问题。
    返回带编号证据供作答引用;证据不足时返回 sufficient=False,请据此向用户拒答。"""
    # ch04 RAG 管线:①Query 理解 → ②混合召回+重排 → ③机械闸 → ④自评闸 → ⑤首尾组装编号
    u = await query_understanding.understand(keyword)
    query = u["standard"]
    # 同义词扩展只拼进检索文本(检索侧),不改标准问法语义
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        search_query, strategy="hybrid_rerank", category=category)

    # 机械闸:无召回 or 最高分低于阈值
    top = hits[0]["rerank_score"] if hits else 0.0
    if not hits or top < settings.rerank_min_score:
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": f"检索证据不足(top={top:.3f})", "citations": []}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"sufficient": False, "source": "self_check",
                "reason": chk["reason"], "citations": []}

    # 首尾组装 + 编号
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"sufficient": True, "evidence": evidence, "citations": citations}

@tool
async def create_ticket(
    description: str,
    ticket_type: Literal["售后", "投诉", "咨询"],
    conversation_id: Annotated[int, InjectedToolArg],
) -> dict:
    """当用户问题需要人工介入(投诉、无法自助解决、明确要求人工)时,创建人工工单。
    description 填用户问题描述,ticket_type 从 售后/投诉/咨询 中选。"""
    ticket_no = await repository.create_ticket(conversation_id, description, ticket_type)
    return {"ticket_no": ticket_no, "status": "已转人工"}


class RefundInput(BaseModel):
    order_id: str = Field(description="要退款的订单号")
    reason: str | None = Field(default=None, description="退款原因(可选,最终以前端固定类目下拉为准)")


@tool(args_schema=RefundInput)
async def submit_refund(order_id: str, reason: str | None = None) -> dict:
    """判定这一单可以退款后,调用本工具发起退款申请。实际提交由前端退款表单确认后落库,
    本工具只表示『这一单可以退,已把提交入口交给用户』。"""
    return {"status": "待用户确认", "order_id": order_id}
