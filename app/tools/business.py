# app/tools/business.py
"""mock 数据源(纯函数),工具实现见 app/tools/builtin/。"""
import random


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
