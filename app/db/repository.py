# app/db/repository.py
from datetime import datetime

from sqlalchemy import case, func, select

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation, Faq, FaithCase, KnowledgeChunk, LowConfidenceQuestion, Message,
    QaExtractionStaging, Ticket,
)

_TICKET_SEQ = 0

def _gen_ticket_no() -> str:
    global _TICKET_SEQ
    _TICKET_SEQ += 1
    return f"T{datetime.now():%Y%m%d%H%M%S}{_TICKET_SEQ:03d}"

async def create_conversation(user_id: str) -> int:
    async with db.async_session() as s:
        conv = Conversation(user_id=user_id)
        s.add(conv)
        await s.commit()
        return conv.id

async def get_conversation(conversation_id: int) -> Conversation | None:
    async with db.async_session() as s:
        return await s.get(Conversation, conversation_id)

async def append_message(
    conversation_id: int,
    role: str,
    content: str | None = None,
    tool_calls: list | None = None,
    tool_call_id: str | None = None,
) -> int:
    async with db.async_session() as s:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        s.add(msg)
        await s.commit()
        return msg.id

async def list_messages(conversation_id: int) -> list[Message]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        )
        return list(result.scalars())

async def search_faq(keyword: str) -> list[Faq]:
    async with db.async_session() as s:
        result = await s.execute(
            select(Faq).where(Faq.question.like(f"%{keyword}%"))
        )
        return list(result.scalars())

async def create_ticket(conversation_id: int, description: str, ticket_type: str) -> str:
    ticket_no = _gen_ticket_no()
    async with db.async_session() as s:
        s.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.status = "已转人工"
        await s.commit()
    return ticket_no

# ---- ch03 知识库:knowledge_chunks / qa_extraction_staging ----

async def insert_knowledge_chunk(
    category: str, questions: str, answer: str,
    section_path: str | None = None, content_type: str | None = None,
    is_key_clause: int = 0,
) -> int:
    async with db.async_session() as s:
        row = KnowledgeChunk(
            category=category, questions=questions, answer=answer,
            section_path=section_path, content_type=content_type,
            is_key_clause=is_key_clause,
        )
        s.add(row)
        await s.commit()
        return row.id

async def list_pending_chunks() -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending")
            .order_by(KnowledgeChunk.id)
        )
        return list(result.scalars())

async def mark_chunk_vectorized(chunk_id: int, vector_id: str) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.vector_id = vector_id
            row.vectorize_status = "done"
            await s.commit()

async def set_chunk_neighbors(chunk_id: int, prev_id: int | None, next_id: int | None) -> None:
    async with db.async_session() as s:
        row = await s.get(KnowledgeChunk, chunk_id)
        if row is not None:
            row.prev_chunk_id = prev_id
            row.next_chunk_id = next_id
            await s.commit()

async def count_chunks_by_status(status: str) -> int:
    async with db.async_session() as s:
        result = await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == status)
        )
        return int(result.scalar_one())

async def list_all_questions() -> list[str]:
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions))
        return list(result.scalars())

async def insert_staging(batch_no: str, source_ref: str | None, question: str, answer: str) -> int:
    async with db.async_session() as s:
        row = QaExtractionStaging(
            batch_no=batch_no, source_ref=source_ref, question=question, answer=answer
        )
        s.add(row)
        await s.commit()
        return row.id

async def list_staging_by_status(status: str) -> list[QaExtractionStaging]:
    async with db.async_session() as s:
        result = await s.execute(
            select(QaExtractionStaging)
            .where(QaExtractionStaging.status == status)
            .order_by(QaExtractionStaging.id)
        )
        return list(result.scalars())

async def set_staging_status(ids: list[int], status: str) -> None:
    if not ids:
        return
    async with db.async_session() as s:
        for i in ids:
            row = await s.get(QaExtractionStaging, i)
            if row is not None:
                row.status = status
        await s.commit()

async def list_conversations_with_messages() -> list[tuple[int, list[Message]]]:
    async with db.async_session() as s:
        conv_ids = list((await s.execute(select(Conversation.id).order_by(Conversation.id))).scalars())
        out = []
        for cid in conv_ids:
            msgs = list((await s.execute(
                select(Message).where(Message.conversation_id == cid).order_by(Message.id)
            )).scalars())
            out.append((cid, msgs))
        return out

# ---- ch03 录入页盘点读数 ----

async def knowledge_stats() -> dict:
    async with db.async_session() as s:
        total = int((await s.execute(
            select(func.count()).select_from(KnowledgeChunk))).scalar_one())
        pending = int((await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "pending"))).scalar_one())
        done = int((await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.vectorize_status == "done"))).scalar_one())
        key_clauses = int((await s.execute(
            select(func.count()).select_from(KnowledgeChunk)
            .where(KnowledgeChunk.is_key_clause == 1))).scalar_one())
        return {"total": total, "pending": pending, "done": done, "key_clauses": key_clauses}

async def list_recent_chunks(limit: int = 20) -> list[KnowledgeChunk]:
    async with db.async_session() as s:
        result = await s.execute(
            select(KnowledgeChunk).order_by(KnowledgeChunk.id.desc()).limit(limit)
        )
        return list(result.scalars())

async def list_chunk_pairs() -> list[tuple[str, str]]:
    """全量 (questions, answer),供录入查重算指纹。"""
    async with db.async_session() as s:
        result = await s.execute(select(KnowledgeChunk.questions, KnowledgeChunk.answer))
        return [(q, a) for q, a in result.all()]

async def staging_stats() -> dict:
    async with db.async_session() as s:
        out = {}
        for st in ("extracted", "kept", "discarded"):
            out[st] = int((await s.execute(
                select(func.count()).select_from(QaExtractionStaging)
                .where(QaExtractionStaging.status == st))).scalar_one())
        return out


async def insert_low_confidence(
    conversation_id: int | None, raw_question: str, source: str, reason: str | None
) -> int:
    """低置信问题落池(ch04:retrieval_low_conf / self_check 两入口;user_feedback 留数据飞轮章)。"""
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


def _faith_dict(row: FaithCase) -> dict:
    return {
        "id": row.id, "eval_id": row.eval_id, "query": row.query, "answer": row.answer,
        "citations": row.citations, "status": row.status, "resolution": row.resolution,
        "seen_count": row.seen_count, "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def upsert_faith_case(
    eval_id: str, query: str, answer: str, citations: list | None
) -> dict:
    """编造个案一题一行:已存在则更新快照并 seen_count+1;已处置过的再判出来
    退回「未解决」(清说明与处置时间)并返回复发标记。"""
    async with db.async_session() as s:
        row = (await s.execute(
            select(FaithCase).where(FaithCase.eval_id == eval_id))).scalar_one_or_none()
        relapsed = False
        if row is None:
            row = FaithCase(eval_id=eval_id, query=query, answer=answer, citations=citations)
            s.add(row)
            await s.flush()
        else:
            row.query = query
            row.answer = answer
            row.citations = citations
            row.seen_count += 1
            row.last_seen_at = datetime.now()
            if row.status != "unresolved":
                row.status = "unresolved"
                row.resolution = None
                row.resolved_at = None
                relapsed = True
        await s.commit()
        return {"id": row.id, "relapsed": relapsed, "seen_count": row.seen_count}


async def list_faith_cases(status: str | None, page: int = 1, size: int = 20) -> dict:
    """台账分页:未解决排前,再按最近出现倒序;带三状态计数(不受筛选影响)。"""
    async with db.async_session() as s:
        counts = {"unresolved": 0, "resolved": 0, "wontfix": 0}
        for st, n in (await s.execute(
                select(FaithCase.status, func.count()).group_by(FaithCase.status))).all():
            counts[st] = int(n)
        conds = [FaithCase.status == status] if status else []
        total = (await s.execute(
            select(func.count()).select_from(FaithCase).where(*conds))).scalar_one()
        rows = (await s.execute(
            select(FaithCase)
            .where(*conds)
            .order_by(case((FaithCase.status == "unresolved", 0), else_=1),
                      FaithCase.last_seen_at.desc(), FaithCase.id.desc())
            .offset((page - 1) * size).limit(size))).scalars().all()
        return {"items": [_faith_dict(r) for r in rows], "total": int(total),
                "page": page, "size": size, "counts": counts}


async def set_faith_case_status(case_id: int, status: str, resolution: str | None = None) -> bool:
    """处置流转:标已解决/无需解决记说明与处置时间(必填校验在 API 层);退回未解决连说明一起清。"""
    async with db.async_session() as s:
        row = await s.get(FaithCase, case_id)
        if row is None:
            return False
        row.status = status
        if status == "unresolved":
            row.resolution = None
            row.resolved_at = None
        else:
            row.resolution = resolution
            row.resolved_at = datetime.now()
        await s.commit()
        return True


async def faith_case_status_map() -> dict[str, str]:
    """eval_id → 处置状态;评估报告侧把本轮判出的个案对回台账状态用。"""
    async with db.async_session() as s:
        rows = (await s.execute(select(FaithCase.eval_id, FaithCase.status))).all()
        return {eid: st for eid, st in rows}


# ---- ch07 会话上下文:摘要读写/计数 ----

async def count_messages_after(conversation_id: int, after_id: int | None) -> int:
    """距上次摘要新增了多少条(after_id 空 = 从未摘要,数全量)。摘要触发判据。"""
    async with db.async_session() as s:
        q = (select(func.count()).select_from(Message)
             .where(Message.conversation_id == conversation_id))
        if after_id:
            q = q.where(Message.id > after_id)
        return int((await s.execute(q)).scalar_one())

async def list_dialog_messages(conversation_id: int) -> list[Message]:
    """user/assistant 消息按 id 升序(摘要任务源数据;tool 行不落库,过滤一道保险)。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.role.in_(("user", "assistant")))
            .order_by(Message.id)
        )
        return list(result.scalars())

async def update_conversation_summary(conversation_id: int, summary: str, upto_msg_id: int) -> None:
    """摘要成功后原子更新两字段(一起写,不会出现摘要新边界旧)。"""
    async with db.async_session() as s:
        conv = await s.get(Conversation, conversation_id)
        if conv is not None:
            conv.summary = summary
            conv.summary_upto_msg_id = upto_msg_id
            await s.commit()
