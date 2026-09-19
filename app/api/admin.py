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


async def _rageval_metrics() -> dict:
    # 只读 rag_eval.json 产物(缺失/写坏都按没跑过),复用 rageval 的读取与派生
    from app.api.rageval import derive_best, load_report
    report = load_report()
    if report is None:
        return {"present": False}
    gen = report.get("generation") or {}
    refusal = (gen.get("refusal") or {}).get("rate")
    return {
        "present": True,
        "best": derive_best(report),
        "best_mrr": ((report.get("retrieval") or {}).get(derive_best(report) or "", {})
                     or {}).get("all", {}).get("mrr"),
        "n_samples": (report.get("meta") or {}).get("n_samples"),
        "refusal_rate": refusal,
    }


async def _observability_metrics() -> dict:
    # ch09:复用观测 API 的三块读取(卡片只要四个数:最烧钱占比/评估轮次/忠实度/在用阈值)
    from app.api.observability import _cal_block, _cost_block, _trend_block
    cost, cal, trend = _cost_block(), _cal_block(), await _trend_block()
    if trend["status"] == "error":
        raise RuntimeError(trend.get("hint") or "eval_runs 读不到")   # mysql 全挂 → 卡片 unreachable
    latest = (trend.get("runs") or [{}])[0] if trend["status"] == "ok" else {}
    metrics = latest.get("metrics") or {}
    return {
        "top_intent": (cost.get("top") or {}).get("intent"),
        "top_share": (cost.get("top") or {}).get("share"),
        "runs": len(trend.get("runs") or []),
        "faithfulness": metrics.get("faithfulness"),
        "current_threshold": cal.get("current_threshold"),
        "recommended_threshold": cal.get("recommended_threshold"),
        "in_sync": cal.get("in_sync"),
    }


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
    ("rageval", "RAG 评估", _rageval_metrics,
     lambda m: ("empty", "还没跑过(评估页可发起)") if not m.get("present")
     else ("ok", f"最佳 {m['best']} · MRR {m['best_mrr']}")),
    ("observability", "观测与成本", _observability_metrics,
     lambda m: ("todo", f"推荐阈值 {m['recommended_threshold']} ≠ 在用 {m['current_threshold']},该回填")
     if m.get("recommended_threshold") is not None and not m.get("in_sync")
     else ("empty", "还没校准(校准作业可发起)") if not m.get("in_sync")
     else ("ok", " · ".join(filter(None, (
         f"忠实度 {m['faithfulness']}" if m.get("faithfulness") is not None else None,
         f"最烧钱 {m['top_intent']} {m['top_share']:.0%}" if m.get("top_share") is not None else None,
     ))) or "评估与成本还没数据(观测页可发起)")),
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
