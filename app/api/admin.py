# app/api/admin.py
"""后台聚合首页:每模块一张卡。某块依赖没起只让它自己那张卡显示读不到,不连坐整页。"""
from fastapi import APIRouter

from app.db.repository import (
    knowledge_stats,
    list_conversations_with_messages,
    staging_stats,
)

router = APIRouter(prefix="/api/admin")


async def _chat_metrics() -> dict:
    convs = await list_conversations_with_messages()
    n_msgs = sum(len(msgs) for _, msgs in convs)
    return {"conversations": len(convs), "messages": n_msgs}


async def _kb_metrics() -> dict:
    return await knowledge_stats()


async def _milvus_metrics() -> dict:
    from app.kb import milvus_client
    client = milvus_client.get_client()
    return {"vectors": milvus_client.count(client)}


async def _staging_metrics() -> dict:
    return await staging_stats()


# (key, 标题, 读数函数, 空数据/要干活的判定与结论)
_CARDS = (
    ("chat", "会话与消息", _chat_metrics,
     lambda m: ("empty", "还没有会话") if m["conversations"] == 0 else ("ok", "服务正常")),
    ("kb", "知识库", _kb_metrics,
     lambda m: ("empty", "还没建库") if m["total"] == 0
     else ("todo", f"待向量化 {m['pending']} 块") if m["pending"] else ("ok", "库存就绪")),
    ("milvus", "向量库", _milvus_metrics,
     lambda m: ("empty", "集合是空的") if m["vectors"] == 0 else ("ok", "可检索")),
    ("staging", "挖知识暂存", _staging_metrics,
     lambda m: ("empty", "暂存表无数据") if sum(m.values()) == 0 else ("ok", "有沉淀待审")),
)


@router.get("/overview")
async def overview() -> dict:
    cards = []
    for key, title, fn, conclude in _CARDS:
        try:
            metrics = await fn()
            state, conclusion = conclude(metrics)
        except Exception:
            metrics, state, conclusion = {}, "unreachable", "读数失败(依赖没起)"
        cards.append({
            "key": key, "title": title,
            "state": state, "conclusion": conclusion, "metrics": metrics,
        })
    return {"cards": cards}
