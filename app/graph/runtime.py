import logging
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.db import repository
from app.graph.build import build_graph

logger = logging.getLogger(__name__)

# 会流式吐答复 token 的节点(其余节点的模型调用 token 不进 delta)
ANSWER_NODES = {"agent_llm"}
# 确定性节点把答复写在 state['answer'],整块作为 delta 吐出
DETERMINISTIC_ANSWER_NODES = {"chitchat_reply", "complaint_reply", "fallback_reply"}

_graph = None
_cm = None  # AsyncSqliteSaver context manager,持有以免被 GC


class ConversationNotFound(Exception):
    pass


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


async def _ensure_conversation(user_id: str, conversation_id: int | None) -> int:
    if conversation_id is None:
        return await repository.create_conversation(user_id)
    if await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)
    return conversation_id


def _graph_input(user_id: str, message: str, cid: int) -> dict:
    # 每轮入口把输出通道清零,防上一轮残留跨轮泄漏(ch05 C1);
    # resume 走 Command(resume=...) 不经此函数,order_id 回填不受影响
    return {"messages": [HumanMessage(message)], "user_id": user_id,
            "conversation_id": cid, "steps": 0, "tokens_used": 0,
            "intent": "", "route": "", "evidence": "", "citations": [],
            "evidence_strong": False, "answer": "", "suggested_actions": [],
            "resolved_query": "", "intent_confidence": 0.0,
            "order_id": "", "order_data": {}}


async def run_turn(user_id, message, conversation_id) -> dict:
    """非流式:落 user 消息 → ainvoke → 返回终态(供 /api/agent、eval)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}
    final = await get_graph().ainvoke(_graph_input(user_id, message, cid), config)
    return {"conversation_id": cid, "state": final}


async def stream_turn(user_id, message, conversation_id) -> AsyncIterator[dict]:
    """流式:落 user 消息 → astream 多模式 → 映射成事件 dict(供 /api/chat 转 SSE)。"""
    cid = await _ensure_conversation(user_id, conversation_id)
    await repository.append_message(cid, "user", content=message)
    config = {"configurable": {"thread_id": str(cid)}}

    actions: list = []
    async for mode, chunk in get_graph().astream(
        _graph_input(user_id, message, cid), config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") in ANSWER_NODES:
                text = msg.content if isinstance(msg.content, str) else ""
                if text:
                    yield {"type": "delta", "text": text}
        elif mode == "updates":
            for node, upd in chunk.items():
                if not isinstance(upd, dict):
                    continue
                if node in DETERMINISTIC_ANSWER_NODES and upd.get("answer"):
                    yield {"type": "delta", "text": upd["answer"]}
                if node == "forced_rag" and upd.get("citations"):
                    yield {"type": "citations", "items": upd["citations"]}
                if node == "agent_tools":
                    for m in upd.get("messages", []):
                        name = getattr(m, "name", None)
                        if name and name != "create_ticket":
                            yield {"type": "tool", "name": name}
                if upd.get("suggested_actions"):
                    actions = upd["suggested_actions"]
    if actions:
        yield {"type": "actions", "items": actions}
    yield {"type": "done", "conversation_id": cid}
