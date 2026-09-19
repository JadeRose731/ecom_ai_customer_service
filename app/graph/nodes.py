import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.types import interrupt

from app.config import settings
from app.core.observability import tag_intent
from app.core import coref, memory
from app.core import intent as intent_mod
from app.core import query_understanding, retrieval, selfcheck
from app.core.llm import get_chat_model
from app.core.prompts import (
    AGENT_SYSTEM, COMPLAINT_REPLY_TEXT, FALLBACK_REPLY_TEXT,
    REFUND_JUDGE_HINT, SCRIPT_REPLY_CHITCHAT, SCRIPT_REPLY_OTHER,
)
from app.db import repository
from app.graph.routing import INTENT_TO_ROUTE
from app.tools import business, engine, registry

logger = logging.getLogger(__name__)

COMPLAINT_REPLY = COMPLAINT_REPLY_TEXT
FALLBACK_REPLY = FALLBACK_REPLY_TEXT


def _user_text(state) -> str:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _history_text(state, max_turns: int = 6) -> str:
    """摘要行 + 滑窗内最近若干轮(不含本轮最后一条 human),供 coref/意图读上下文。
    ch07:先按摘要边界切窗,再取尾——跨滑窗的指代靠摘要行兜住。"""
    msgs = memory.build_window(state.get("messages", []),
                               state.get("summary_upto_msg_id") or 0,
                               settings.context_window_max_tokens)
    prior = msgs[:-1] if msgs else []
    lines = []
    for m in prior[-max_turns:]:
        role = "用户" if isinstance(m, HumanMessage) else "客服"
        text = m.content if isinstance(m.content, str) else ""
        if text:
            lines.append(f"{role}:{text}")
    body = "\n".join(lines)
    head = memory.summary_line(state.get("summary"))
    return f"{head}\n{body}".strip() if head else body


async def complaint_reply(state) -> dict:
    """投诉:安抚话术 + 转人工/建工单两个可选项(后端不执行,交前端自选)。"""
    actions = [
        {"type": "transfer_human"},
        {"type": "create_ticket",
         "draft": {"description": _user_text(state), "ticket_type": "投诉"}},
    ]
    return {"answer": COMPLAINT_REPLY, "suggested_actions": actions,
            "trace": {"route": "escalate"}}


async def fallback_reply(state) -> dict:
    """置信度兜底:证据弱回兜底话术,并把问题落低置信池留给数据飞轮。"""
    reason = f"检索证据不足(top={state.get('trace', {}).get('evidence_top', 0):.3f})"
    await repository.insert_low_confidence(
        state.get("conversation_id"), _user_text(state), "retrieval_low_conf", reason
    )
    return {"answer": FALLBACK_REPLY, "trace": {"route": "fallback"}}


async def script_reply(state) -> dict:
    """闲聊/其他 兜底话术(零模型):按 intent 分文案——闲聊把话题引回产品,其他请用户说具体些。
    分流后立即命中、不进 Agent。(与 ch05 知识路证据弱的 fallback_reply 是两码事,勿混。)"""
    text = SCRIPT_REPLY_OTHER if state.get("intent") == "其他" else SCRIPT_REPLY_CHITCHAT
    return {"answer": text, "trace": {"route": "fallback_script"}}


async def resolve_reference(state) -> dict:
    """指代消解 + Query 改写(合一):吃最近几轮历史把半截话补成完整问句;已完整则透传。
    写 resolved_query,供 classify_intent 与 refund_flow 检索共用(意图识别不再改写)。"""
    query = _user_text(state)
    resolved = await coref.resolve(query, _history_text(state))
    mode = "rewrite" if resolved != query else "passthrough"
    return {"resolved_query": resolved, "trace": {"coref": mode}}


async def classify_intent(state) -> dict:
    """意图四件套:八类 + confidence + 「其他」兜底。吃 resolved_query(空则回落原话)+ 最近历史
    判当前意图(应对物流→退款→物流漂移)。把归属出口 route 落进 State(供 _agent_messages 判注入、
    log 留痕;route_by_intent 条件边按同一 INTENT_TO_ROUTE 分流,单一来源不漂移)。"""
    query = state.get("resolved_query") or _user_text(state)
    r = await intent_mod.classify(query, _history_text(state))
    intent, conf = r["intent"], r["confidence"]
    route = INTENT_TO_ROUTE.get(intent, "business")
    tag_intent(intent, conf, session_id=state.get("conversation_id"))   # ch09:意图进 trace,Cost Control 按意图分堆
    return {"intent": intent, "intent_confidence": conf, "route": route,
            "trace": {"intent": intent, "intent_confidence": conf, "route": route}}


_ORDER_RE = re.compile(r"(?<!\d)(\d{4,})(?!\d)")   # 4+ 位连续数字视作订单号(中文邻接数字无 \b 词边界,\b 取不到)


def _extract_order_id(text: str) -> str | None:
    m = _ORDER_RE.search(text or "")
    return m.group(1) if m else None


async def fetch_order(state) -> dict:
    """退款子流程第一步:抽订单号;缺则 interrupt 弹订单选择器等前端点选(resume 回填);
    拿到后 order_snapshot 取订单数据。interrupt 之前只做只读(resume 时本节点从头重跑)。"""
    oid = state.get("order_id") or _extract_order_id(
        state.get("resolved_query") or _user_text(state))
    if not oid:
        orders = business.list_user_orders(state.get("user_id", ""))   # 只读,可安全重跑
        oid = interrupt({"type": "select_order", "orders": orders})    # resume 回填订单号
    data = business.order_snapshot(oid)
    return {"order_id": oid, "order_data": data,
            "trace": {"fetch_order": {"order_id": oid}}}


async def retrieve_policy(state) -> dict:
    """退款子流程强制检索政策:Query 扩写 3 条 → 多 query 各检索一次 → 按 chunk id 去重合并
    (保留每 id 最高分)→ 按分降序拼编号证据。产出注入 main_agent 作「能不能退」的判据(README L194:
    不让模型凭记忆答)。库里知识一份,扩写只在检索侧现查现用。"""
    base = state.get("resolved_query") or _user_text(state)
    od = state.get("order_data") or {}
    seed = f"{base} {od.get('status', '')}".strip()
    queries = await query_understanding.expand_queries(seed)

    merged: dict = {}                       # chunk id -> 最高分 hit
    for q in queries:
        for h in await retrieval.search_knowledge(q, strategy="hybrid_rerank"):
            cur = merged.get(h["id"])
            if cur is None or h["rerank_score"] > cur["rerank_score"]:
                merged[h["id"]] = h
    ranked = sorted(merged.values(), key=lambda h: h["rerank_score"], reverse=True)
    arranged = retrieval.arrange_head_tail(ranked)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence": evidence, "citations": citations,
            "trace": {"retrieve_policy": {"queries": queries, "hits": len(ranked)}}}


async def forced_rag(state) -> dict:
    """知识类强制检索(复用 ch03/04 检索器):产出编号证据 + 证据强弱信号。
    复刻 query_faq 的两道生成前证据闸(检索分闸 + 自评闸),但把结果落进 State。"""
    query_raw = _user_text(state)
    u = await query_understanding.understand(query_raw)
    query = u["standard"]
    # 同义词扩展只拼进检索文本(检索侧),不改标准问法语义——与 query_faq 同一惯例
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(search_query, strategy="hybrid_rerank")
    top = hits[0]["rerank_score"] if hits else 0.0

    # 机械闸:无召回 or 最高分低于阈值
    if not hits or top < settings.rerank_min_score:
        return {"evidence_strong": False,
                "trace": {"forced_rag": True, "evidence_top": top}}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"evidence_strong": False,
                "trace": {"forced_rag": True, "evidence_top": top, "self_check": chk["reason"]}}

    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"evidence_strong": True, "evidence": evidence, "citations": citations,
            "trace": {"forced_rag": True, "evidence_top": top}}


async def confidence_check(state) -> dict:
    """置信度兜底(生成前证据闸)的实体节点:留痕门控决策供可观测;
    强/弱的实际分流在其后的条件边 confidence_gate(读 evidence_strong)。"""
    decision = "strong" if state.get("evidence_strong") else "weak"
    return {"trace": {"confidence": decision}}


_KNOWLEDGE_EVIDENCE_HINT = (
    "\n\n## 已检索到的知识证据(请据此作答,每个关键结论后标注来源编号如[1];"
    "证据已给,不要再调用 query_faq;仍可按需调用订单/物流等工具)\n"
)


TURN_CTX_ID = "turn-ctx"   # 哨兵:日志据此把注入块跟真实用户消息分开。只能用 id 不能用 name
                           # ——name 会被序列化发给上游,改变请求字节;id 不会。


def _turn_context(state) -> str:
    """本轮才有的材料:早前摘要 + 检索证据 + 退款路的订单数据。没有就返回空串。

    摘要排最前(它讲的是更早发生的事);顺序不能反:REFUND_JUDGE_HINT 的措辞是
    「下面给出该订单数据与检索到的退换货政策证据」,它假设证据已在前文给过。"""
    parts = []
    ss = memory.summary_system(state.get("summary"))
    if ss is not None:
        parts.append("\n\n" + ss.content)
    if state.get("evidence"):
        parts.append(_KNOWLEDGE_EVIDENCE_HINT + state["evidence"])
    if state.get("route") == "refund_flow":
        parts.append(REFUND_JUDGE_HINT + json.dumps(state.get("order_data", {}), ensure_ascii=False))
    return "".join(parts)     # 三段文本逐字沿用原来的常量,一个字都不改:这次只挪位置


def _with_turn_context(window: list, turn_ctx: str) -> list:
    """插在滑窗里最后一条用户消息之后:ReAct 第 2 步的 [Sys,问,材料,AI,Tool] 要完整
    包含第 1 步的 [Sys,问,材料] 作前缀,轮内才命中缓存。窗内无用户消息则退化成追加。"""
    msg = HumanMessage(turn_ctx, id=TURN_CTX_ID)
    for i in range(len(window) - 1, -1, -1):
        if isinstance(window[i], HumanMessage):
            return [*window[:i + 1], msg, *window[i + 1:]]
    return [*window, msg]


def _agent_messages(state) -> list:
    """拼装顺序固定(ch07):人设+红线 system → 滑窗原文 → 摘要与本轮材料(紧跟用户那句)。

    整条消息列表里**只有一条 SystemMessage**,内容恒为 AGENT_SYSTEM。这不是洁癖:
    上游的 chat template 会把列表里所有 system 消息上提、合并成一个头部块渲染,所以
    「摘要单独放第二条 system」等于把它拼在了 AGENT_SYSTEM 后面,工具 schema 被挤到
    可变内容之后 —— 而 prompt caching 按渲染后的前缀精确匹配,于是整段前缀全 miss。
    实测:无摘要的会话 cache_read=2048,摘要一进 system 就掉到 0。
    摘要因此跟证据一起走用户侧那条消息。

    滑窗=摘要边界锚点切 + trim_messages token 兜底;State 全量历史不动,这里只现拼精简版。"""
    window = memory.build_window(state.get("messages", []),
                                 state.get("summary_upto_msg_id") or 0,
                                 settings.context_window_max_tokens)
    turn_ctx = _turn_context(state)
    if turn_ctx:
        window = _with_turn_context(window, turn_ctx)
    return [SystemMessage(AGENT_SYSTEM), *window]


def _log_model_context(state, msgs) -> None:
    """验收可观测:main_agent 调模型前打 model_ctx 日志——摘要全文 + 滑窗每条(角色+前40字)
    + 窗口条数 + token 估算。两件套之一(另一件是前端会话侧栏),拼装结果据此肉眼可查。"""
    summary = state.get("summary") or ""
    head = (f"model_ctx conv={state.get('conversation_id')} msgs={len(msgs)} "
            f"~{count_tokens_approximately(msgs)}tok summary={summary!r}")
    lines = [head]
    for m in msgs:
        text = m.content if isinstance(m.content, str) else ""
        lines.append(f"  [{m.type}] '{text[:40]}'")
    logger.info("\n".join(lines))


async def agent_llm(state, config=None) -> dict:
    """ReAct 推理步:调模型(带工具),累加 steps 与 token 消耗。"""
    tag_intent(state.get("intent", "其他"), state.get("intent_confidence", 0.0),
               session_id=state.get("conversation_id"))   # ch09:本节点 LLM span 按 intent 分组(节点 task 边界,见 dev-notes)
    msgs = _agent_messages(state)
    _log_model_context(state, msgs)   # ch07:拼装结果进日志,验收可观测
    # ch08:全量清单现问现拿(bind),MCP 新工具下一轮即可见
    specs = await registry.get_all_specs()
    model = get_chat_model(streaming=True).bind_tools([s.tool for s in specs])
    ai: AIMessage = await model.ainvoke(msgs, config)
    used = (ai.usage_metadata or {}).get("total_tokens", 0) if ai.usage_metadata else 0
    return {"messages": [ai],
            "steps": state.get("steps", 0) + 1,
            "tokens_used": state.get("tokens_used", 0) + used}


async def agent_tools(state) -> dict:
    """ReAct 行动步(ch08):一切工具经统一执行引擎。create_ticket(唯一写操作)走确认流——
    参数齐则节点顶部 interrupt 推工单预览(interrupt 前只做纯计算,resume 重跑安全,照 fetch_order 范式);
    参数缺则交引擎按「校验拦下」回灌,模型自然向用户追问。submit_refund 仍拦成前端退款表单(ch06)。"""
    last = state["messages"][-1]
    cid = state.get("conversation_id", 0)
    specs = {s.name: s for s in await registry.get_all_specs()}

    ticket_calls = [tc for tc in last.tool_calls if tc["name"] == "create_ticket"]
    decision = None
    tspec = specs.get("create_ticket")
    if ticket_calls and tspec is not None \
            and engine.validate_args(tspec, dict(ticket_calls[0].get("args") or {})) is None:
        first_args = ticket_calls[0]["args"]
        decision = interrupt({"type": "confirm_ticket",
                              "preview": {"ticket_type": first_args.get("ticket_type", "咨询"),
                                          "description": first_args.get("description", "")}})

    tool_msgs = []
    actions = list(state.get("suggested_actions", []))
    for tc in last.tool_calls:
        if tc["name"] == "submit_refund":
            actions.append({"type": "refund_form",
                            "draft": {"order_id": tc["args"].get("order_id", ""),
                                      "reason": tc["args"].get("reason")}})
            tool_msgs.append(ToolMessage(
                content="已把『提交退款工单』选项交给用户确认。请用一句话说明这一单可以退款并停止,不要再调用任何工具。",
                tool_call_id=tc["id"], name="submit_refund"))
        elif tc["name"] == "create_ticket" and decision is not None:
            if tc is ticket_calls[0]:
                if decision.get("confirmed"):
                    run = await engine.execute_tool_call(tc, cid, specs, confirmed=True)
                else:
                    run = await engine.execute_tool_call(
                        tc, cid, specs, confirmed=False,
                        deny_note="用户在工单预览卡片上点了取消,本次不建单。请勿再发起,除非用户再次明确要求。")
                tool_msgs.append(run.tool_message)
            else:
                tool_msgs.append(ToolMessage(content="一次只处理一个建工单请求,本次调用已忽略。",
                                             tool_call_id=tc["id"], name="create_ticket", status="error"))
        else:
            run = await engine.execute_tool_call(tc, cid, specs)
            tool_msgs.append(run.tool_message)
    out = {"messages": tool_msgs}
    if actions:
        out["suggested_actions"] = actions
    return out


def resolve_answer(state) -> str:
    """最终答复:确定性节点写在 state['answer'];Agent 答复取最后一条 AIMessage 文本。"""
    if state.get("answer"):
        return state["answer"]
    for m in reversed(state.get("messages", [])):
        if isinstance(m, AIMessage):
            content = m.content
            if isinstance(content, str):
                return content
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def log_node(state) -> dict:
    """日志记录:留痕 intent/route/trace(可观测地基),并落 MySQL 一条 assistant 消息(审计)。"""
    logger.info(
        "ch05 turn conv=%s intent=%s route=%s trace=%s",
        state.get("conversation_id"), state.get("intent"), state.get("route"),
        state.get("trace", {}),
    )
    answer = resolve_answer(state)
    if state.get("conversation_id"):
        await repository.append_message(state["conversation_id"], "assistant",
                                        content=answer or None)
    return {}
