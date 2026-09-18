import json
import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.config import settings
from app.core import summarizer
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"agent_llm"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"script_reply", "complaint_reply", "fallback_reply"}
# 会产出 citations 的检索节点(ch06 新增 retrieve_policy)
CITATION_NODES = {"forced_rag", "retrieve_policy"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


def dedup_actions(actions: list) -> list:
    """按 (type, draft) 去重 keep-first:同型同稿的建议动作只留首条。"""
    seen: set = set()
    out: list = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        key = (a.get("type"),
               json.dumps(a.get("draft"), ensure_ascii=False, sort_keys=True,
                          default=str))
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


async def init_graph() -> None:
    """lifespan 起:开持久 checkpointer + setup + 编译图。"""
    global _graph, _cm
    _cm = AsyncSqliteSaver.from_conn_string(settings.checkpointer_db_path)
    checkpointer = await _cm.__aenter__()
    await checkpointer.setup()
    _graph = build_graph(checkpointer=checkpointer)
    logger.info("ch05 图已编译,checkpointer=%s", settings.checkpointer_db_path)


async def close_graph() -> None:
    global _graph, _cm
    if _cm is not None:
        await _cm.__aexit__(None, None, None)
        _cm = None
    _graph = None


def get_graph():
    if _graph is None:
        raise RuntimeError("图未初始化,应在 FastAPI lifespan 调 init_graph()")
    return _graph


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> tuple[int, str, int]:
    """返回 (cid, summary, summary_upto_msg_id):入口一次库读把摘要两字段一并带出。"""
    if conversation_id is None:
        return await repository.create_conversation(user_id), "", 0
    conv = await repository.get_conversation(conversation_id)
    if conv is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id, conv.summary or "", conv.summary_upto_msg_id or 0


def _graph_input(user_id: str, message: str, cid: int, msg_id: int,
                 summary: str, summary_upto: int) -> dict:
    # 每轮入口把输出通道清零,防上一轮残留跨轮泄漏(ch05 C1);
    # resume 走 Command(resume=...) 不经此函数,order_id 回填不受影响。
    # ch07:用户消息带 db-{msg_id} 锚点(摘要边界对齐);summary 两字段每轮刷新。
    return {"messages": [HumanMessage(message, id=f"db-{msg_id}")], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "intent": "", "route": "", "evidence": "", "citations": [],
            "evidence_strong": False, "answer": "", "suggested_actions": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {},
            "summary": summary, "summary_upto_msg_id": summary_upto,
            "trace": None}


def _interrupt_orders(state: dict):
    """ainvoke 终态里的待处理中断 → 订单选择器载荷(无中断返回 None)。"""
    intr = state.get("__interrupt__")
    if not intr:
        return None
    return intr[0].value.get("orders")


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。"""
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}
    final = await get_graph().ainvoke(
        _graph_input(user_id, message, cid, msg_id, summary, upto), config)
    await summarizer.maybe_schedule_summary(cid)     # 轮后触发检查(后台,不阻塞返回)
    return {"conversation_id": cid, "state": final, "interrupt": _interrupt_orders(final)}


async def resume_turn(conversation_id: int, resume_value) -> dict:
    """非流式续跑(供 /api/agent、eval):Command(resume) 回填后跑到下一个中断或结束。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    config = {"configurable": {"thread_id": str(conversation_id)}}
    final = await get_graph().ainvoke(Command(resume=resume_value), config)
    await summarizer.maybe_schedule_summary(conversation_id)
    return {"conversation_id": conversation_id, "state": final,
            "interrupt": _interrupt_orders(final)}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 映射成事件 dict(供 /api/chat 转 SSE)。"""
    cid, summary, upto = await _ensure_conversation(user_id, conversation_id)
    msg_id = await repository.append_message(cid, "user", content=message)
    async for ev in _stream_events(cid, _graph_input(user_id, message, cid, msg_id, summary, upto)):
        yield ev
    await summarizer.maybe_schedule_summary(cid)


async def stream_resume(conversation_id: int, resume_value) -> AsyncIterator[dict]:
    """前端点选订单后续跑同一会话的暂停流。"""
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    async for ev in _stream_events(conversation_id, Command(resume=resume_value)):
        yield ev
    await summarizer.maybe_schedule_summary(conversation_id)


async def _stream_events(cid: int, stream_source) -> AsyncIterator[dict]:
    config = {"configurable": {"thread_id": str(cid)}}
    actions: list = []
    async for mode, chunk in get_graph().astream(
        stream_source, config, stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                # R15:中断流无 done 帧,conversation_id 必须随 interrupt 帧下发,
                # 否则全新会话首问即中断时前端拿不到 cid、无从 resume
                yield {"type": "interrupt", "kind": payload.get("type", ""),
                       "orders": payload.get("orders", []), "conversation_id": cid}
                return  # 图已暂停,结束本次流(前端点选后走 resume 续流)
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node in CITATION_NODES and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name not in ("create_ticket", "submit_refund"):
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions.extend(upd["suggested_actions"])
    if actions:
        yield {"type": "actions", "items": dedup_actions(actions)}
    yield {"type": "done", "conversation_id": cid}
