# app/core/agent.py
import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.memory import trim_history
from app.core.prompts import AGENT_SYSTEM, RAG_ANSWER_SYSTEM
from app.db import repository
from app.db.models import Message
from app.tools.infra import ToolRun, execute_tool_call
from app.tools.registry import get_all_tools

class ConversationNotFound(Exception):
    pass

@dataclass
class AgentResult:
    conversation_id: int
    answer: str
    tool_calls: list[dict]
    tool_runs: list[ToolRun]
    citations: list = field(default_factory=list)   # ch04:query_faq 命中时的编号引用

def _text(msg: AIMessage) -> str:
    if isinstance(msg.content, str):
        return msg.content
    # content 为分块 list 时拼接文本片段
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))

def _build_history(rows: list[Message]) -> list[BaseMessage]:
    """跨轮上下文:只回放 user 与「content 非空且非工具调用」的 assistant。
    带 tool_calls 的 assistant(常夹带 preamble)不跨轮回放,避免污染续接。"""
    out: list[BaseMessage] = [SystemMessage(AGENT_SYSTEM)]
    for m in rows:
        if m.role == "user":
            out.append(HumanMessage(m.content or ""))
        elif m.role == "assistant" and m.content and not m.tool_calls:
            out.append(AIMessage(m.content))
    return trim_history(out, max_tokens=settings.token_budget)

async def _prepare_turn(user_id, message, conversation_id, model):
    """共享编排前半:会话身份 → 落 user → 组装历史 → turn1 bind_tools 定工具 →
    落 assistant → 有工具则并行执行落 tool。返回 (cid, messages, ai, runs)。"""
    if conversation_id is None:
        conversation_id = await repository.create_conversation(user_id)
    elif await repository.get_conversation(conversation_id) is None:
        raise ConversationNotFound(conversation_id)

    await repository.append_message(conversation_id, "user", content=message)
    rows = await repository.list_messages(conversation_id)
    messages = _build_history(rows)

    ai: AIMessage = await model.bind_tools(get_all_tools()).ainvoke(messages)
    await repository.append_message(
        conversation_id, "assistant",
        content=_text(ai) or None, tool_calls=ai.tool_calls or None,
    )

    runs: list[ToolRun] = []
    if ai.tool_calls:
        runs = await asyncio.gather(
            *(execute_tool_call(tc, conversation_id) for tc in ai.tool_calls)
        )
        for r in runs:
            await repository.append_message(
                conversation_id, "tool",
                content=r.tool_message.content, tool_call_id=r.tool_call_id,
            )
    return conversation_id, messages, ai, runs

def _faq_result(runs) -> dict | None:
    """取 query_faq 的结果 JSON(证据足→citations;不足→落池信号)。无/解析失败回 None。"""
    for r in runs:
        if getattr(r, "name", None) == "query_faq":
            try:
                return json.loads(r.tool_message.content)
            except (ValueError, TypeError):
                return None
    return None


def _convergence_messages(messages: list[BaseMessage], ai: AIMessage, runs: list[ToolRun],
                          faq: dict | None) -> list[BaseMessage]:
    """收敛步消息:默认沿用 agent system;query_faq 证据足时换用带引用规则 +
    负面知识的 RAG system(spec §1.3 生成侧),答案才会带 [n] 角标。"""
    conv = [*messages, ai, *(r.tool_message for r in runs)]
    if faq is not None and faq.get("sufficient"):
        return [SystemMessage(RAG_ANSWER_SYSTEM), *conv[1:]]
    return conv


async def run_agent_turn(user_id, message, conversation_id, model=None) -> AgentResult:
    """非流式出口:一次性返回工具轨迹 + 最终回答(供 /api/agent、eval、单测)。"""
    model = model or get_chat_model()
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)
    if not ai.tool_calls:
        return AgentResult(conversation_id, _text(ai), [], [])
    faq = _faq_result(runs)
    citations: list = []
    if faq is not None:
        if faq.get("sufficient"):
            citations = faq.get("citations", [])
        else:  # 证据不足:生成前先落池(确定性,不依赖模型)
            await repository.insert_low_confidence(
                conversation_id, message, faq.get("source", "self_check"), faq.get("reason"))
    final: AIMessage = await model.ainvoke(         # 收敛:不 bind_tools
        _convergence_messages(messages, ai, runs, faq))
    await repository.append_message(conversation_id, "assistant", content=_text(final))
    return AgentResult(conversation_id, _text(final), ai.tool_calls, runs, citations)

async def stream_agent_turn(user_id, message, conversation_id, model=None) -> AsyncIterator[dict]:
    """流式出口:产出事件 dict 供 /api/chat 转 SSE。收敛用 astream 逐 token、不 bind。"""
    model = model or get_chat_model(streaming=True)
    conversation_id, messages, ai, runs = await _prepare_turn(
        user_id, message, conversation_id, model)

    if not ai.tool_calls:
        # 无工具:turn1 文本即答案(assistant 已落库),整块吐出
        yield {"type": "delta", "text": _text(ai)}
        yield {"type": "done", "conversation_id": conversation_id}
        return

    for tc in ai.tool_calls:
        yield {"type": "tool", "name": tc.get("name") or ""}

    # ch04:工具轨迹帧后、收敛生成前——证据足先推 citations(前端挂引用数据),
    # 不足则落池(拒答信号已在 tool 结果里,模型据此显式拒答)
    faq = _faq_result(runs)
    if faq is not None:
        if faq.get("sufficient"):
            yield {"type": "citations", "items": faq.get("citations", [])}
        else:
            await repository.insert_low_confidence(
                conversation_id, message, faq.get("source", "self_check"), faq.get("reason"))

    chunks: list[str] = []
    async for chunk in model.astream(_convergence_messages(messages, ai, runs, faq)):
        text = chunk.content if isinstance(chunk.content, str) else _text(chunk)
        if not text:
            continue
        chunks.append(text)
        yield {"type": "delta", "text": text}

    await repository.append_message(conversation_id, "assistant", content="".join(chunks))
    yield {"type": "done", "conversation_id": conversation_id}
