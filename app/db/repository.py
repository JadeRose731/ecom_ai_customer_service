# app/db/repository.py
from datetime import datetime

from sqlalchemy import case, delete, func, select, update

import app.db.base as db          # 用模块属性引用,便于测试 monkeypatch async_session
from app.db.models import (
    Conversation, EvalRun, Faq, FaithCase, KnowledgeChunk, LowConfidenceQuestion, Message,
    QaExtractionStaging, ReviewQueue, Ticket, ToolAuditLog, TopicClassification,
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


async def delete_pending_chunks_by_section(section_path: str) -> int:
    """清某 section 的 pending 残件。审核写回失败后重试前必调——vectorize_pending 捞全部 pending,
    不清会把上次失败行和新行一起送进 Milvus,知识库从此检索出重复证据(ch09 review)。"""
    async with db.async_session() as s:
        result = await s.execute(
            delete(KnowledgeChunk).where(
                KnowledgeChunk.section_path == section_path,
                KnowledgeChunk.vectorize_status == "pending",
            )
        )
        await s.commit()
        return result.rowcount

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
    conversation_id: int | None, raw_question: str, source: str, reason: str | None,
    retrieved_chunks: list | None = None,
) -> int:
    """低置信问题落池(ch04:retrieval_low_conf / self_check 两入口;user_feedback 留数据飞轮章)。
    ch09:retrieved_chunks 带召回快照(Top3 原文+得分),走了检索才写,没走为 NULL。"""
    async with db.async_session() as s:
        row = LowConfidenceQuestion(
            conversation_id=conversation_id, raw_question=raw_question,
            source=source, reason=reason, retrieved_chunks=retrieved_chunks,
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


# ---- ch09 数据飞轮(待审队列 + 评估轮次)----


async def fetch_unmatched_low_conf(limit: int) -> list[LowConfidenceQuestion]:
    """飞轮处理游标:尚未归并(matched_review_id IS NULL)的池内问题,按 id 升序。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id.is_(None))
            .order_by(LowConfidenceQuestion.id).limit(limit)
        )
        return list(result.scalars())


async def list_review_candidates(limit: int = 200) -> list[dict]:
    """查重候选:全部状态的缺口行(用户拍板:比全部,驳回即终审),updated_at 倒序截断防 token 爆。"""
    async with db.async_session() as s:
        result = await s.execute(
            select(ReviewQueue.id, ReviewQueue.normalized_question)
            .order_by(ReviewQueue.updated_at.desc()).limit(limit)
        )
        return [{"id": r.id, "normalized_question": r.normalized_question} for r in result]


async def insert_review_item(normalized_question: str, ai_suggested_answer: str | None) -> int:
    async with db.async_session() as s:
        row = ReviewQueue(normalized_question=normalized_question,
                          ai_suggested_answer=ai_suggested_answer)
        s.add(row)
        await s.commit()
        return row.id


async def increment_occurrence(review_id: int) -> None:
    """查重命中:只累加次数,状态不动(命中已驳回/已通过也一样——驳回即终审)。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is not None:
            row.occurrence_count = row.occurrence_count + 1
            await s.commit()


async def set_matched_review(lcq_id: int, review_id: int) -> None:
    async with db.async_session() as s:
        row = await s.get(LowConfidenceQuestion, lcq_id)
        if row is not None:
            row.matched_review_id = review_id
            await s.commit()


async def list_review_queue(status: str | None) -> list[ReviewQueue]:
    """审核页列表:按出现次数降序(频次=优先级),同频新的在前。"""
    async with db.async_session() as s:
        q = select(ReviewQueue).order_by(
            ReviewQueue.occurrence_count.desc(), ReviewQueue.id.desc())
        if status:
            q = q.where(ReviewQueue.review_status == status)
        return list((await s.execute(q)).scalars())


async def get_review_detail(review_id: int) -> tuple[ReviewQueue, list[LowConfidenceQuestion]] | None:
    """详情:缺口行 + 归并进来的原话流水(带 source/快照,审核人判断真缺还是没检到)。"""
    async with db.async_session() as s:
        item = await s.get(ReviewQueue, review_id)
        if item is None:
            return None
        raws = list((await s.execute(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id)
        )).scalars())
        return item, raws


async def update_review_status(review_id: int, status: str, approved_answer: str | None = None) -> bool:
    """仅「待审」可流转(通过/驳回都是终态);返回是否真的更新了。"""
    async with db.async_session() as s:
        row = await s.get(ReviewQueue, review_id)
        if row is None or row.review_status != "待审":
            return False
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer
        await s.commit()
        return True


async def claim_review(review_id: int, approved_answer: str) -> bool:
    """原子占位(条件 UPDATE 在库上判「待审」):审核通过先占位再写知识库——
    先写后改状态的话,并发双击/重试会各自插一份同内容 chunk(ch09 review)。"""
    async with db.async_session() as s:
        result = await s.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id, ReviewQueue.review_status == "待审")
            .values(review_status="通过", approved_answer=approved_answer)
        )
        await s.commit()
        return result.rowcount == 1


async def revert_review_claim(review_id: int) -> None:
    """占位后写回失败的补偿:退回待审并清核准答案。只有占位成功者会调,无条件回退安全。"""
    async with db.async_session() as s:
        await s.execute(
            update(ReviewQueue)
            .where(ReviewQueue.id == review_id)
            .values(review_status="待审", approved_answer=None)
        )
        await s.commit()


async def insert_eval_run(triggered_by: str, dataset_size: int, metrics: dict) -> int:
    async with db.async_session() as s:
        row = EvalRun(triggered_by=triggered_by, dataset_size=dataset_size, metrics=metrics)
        s.add(row)
        await s.commit()
        return row.id


async def list_eval_runs(limit: int = 10) -> list[EvalRun]:
    async with db.async_session() as s:
        result = await s.execute(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit))
        return list(result.scalars())


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

async def list_conversations(user_id: str, limit: int = 50) -> list[dict]:
    """某用户的会话列表(新在前,带首问预览 + 有无摘要标记),前端多会话切换用。"""
    async with db.async_session() as s:
        rows = (await s.execute(
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc())
            .limit(limit))).scalars().all()
        out = []
        for c in rows:
            first = (await s.execute(
                select(Message.content)
                .where(Message.conversation_id == c.id, Message.role == "user")
                .order_by(Message.id)
                .limit(1))).scalar_one_or_none()
            out.append({
                "id": c.id, "status": c.status,
                "preview": (first or "")[:40],
                "has_summary": bool(c.summary),
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
            })
        return out

# ---- ch08 工具审计 ----


async def insert_tool_audit(
    conversation_id: int | None,
    tool_call_id: str | None,
    tool_name: str,
    tool_source: str,
    mcp_server: str | None,
    arguments: dict | None,
    result_summary: str | None,
    status: str,
    error_message: str | None,
    retry_count: int,
    duration_ms: int | None,
) -> None:
    """工具调用审计落一条。调用方(engine)自行 try/except——审计失败不许反拦工具执行。"""
    async with db.async_session() as s:
        s.add(ToolAuditLog(
            conversation_id=conversation_id, tool_call_id=tool_call_id,
            tool_name=tool_name, tool_source=tool_source, mcp_server=mcp_server,
            arguments=arguments, result_summary=result_summary, status=status,
            error_message=error_message, retry_count=retry_count, duration_ms=duration_ms,
        ))
        await s.commit()


# ---------- ch10 主题分类 ----------

def _pool_text_stmt():
    """池问题取文本的公共查询:优先归并后的标准化问法,没归并退回原话。"""
    return (
        select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
        .order_by(LowConfidenceQuestion.id)
    )


async def list_pool_texts() -> list[dict]:
    """ch10 训练语料捞取:全量低置信度问题。"""
    async with db.async_session() as s:
        rows = (await s.execute(_pool_text_stmt())).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def list_unclassified_questions(limit: int = 500) -> list[dict]:
    """ch10 旁路批处理捞取:尚未归类、且已归并的低置信度问题。
    只归有 matched_review_id 的问题——分类器只吃归并阶段产出的标准化问法,
    未归并的留到下一轮归并后再归,不喂原话。"""
    stmt = (
        _pool_text_stmt()
        .outerjoin(TopicClassification,
                   TopicClassification.question_id == LowConfidenceQuestion.id)
        .where(TopicClassification.id.is_(None))
        .where(LowConfidenceQuestion.matched_review_id.is_not(None))
        .limit(limit)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    return [{"question_id": qid, "text": norm or raw} for qid, raw, norm in rows]


async def insert_topic_classifications(rows: list[dict]) -> int:
    """批量写归类结果;rows: [{question_id, labels}]。"""
    async with db.async_session() as s:
        s.add_all([TopicClassification(question_id=r["question_id"], labels=r["labels"])
                   for r in rows])
        await s.commit()
    return len(rows)


async def topic_distribution(samples_per_class: int = 3) -> dict:
    """主题分布:17 类各自问题量 + 每类样例;labels JSON 在 Python 侧聚合(量级小)。"""
    from app.core.taxonomy import TOPIC_NAMES

    stmt = (
        select(TopicClassification.labels, LowConfidenceQuestion.raw_question,
               ReviewQueue.normalized_question, TopicClassification.classified_at)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    counts = {name: 0 for name in TOPIC_NAMES}
    samples: dict[str, list[str]] = {name: [] for name in TOPIC_NAMES}
    seen: dict[str, set[str]] = {name: set() for name in TOPIC_NAMES}
    latest = None
    for labels, raw, norm, ts in rows:
        text = norm or raw
        latest = ts if latest is None or ts > latest else latest
        for lb in labels or []:
            if lb in counts:
                counts[lb] += 1
                # 样例按展示文本去重:归并后同一句问法对应池里好几行,重复样例白给
                if text not in seen[lb] and len(samples[lb]) < samples_per_class:
                    seen[lb].add(text)
                    samples[lb].append(text)
    return {
        "total": len(rows),
        "latest": latest.isoformat() if latest else None,
        "classes": [{"label": n, "count": counts[n], "samples": samples[n]}
                    for n in TOPIC_NAMES],
    }


async def topic_questions(label: str, page: int = 1, size: int = 20) -> dict:
    """类目问题列表:labels 命中判断与分页在 Python 侧做(池量级百级,不依赖 MySQL JSON 函数)。
    未归并显示原话;归并显示标准化问法并另带原话;审核状态跟着 review_queue;
    附来源/同义合并条数/归类时间供列表页展示。"""
    stmt = (
        select(TopicClassification.question_id, TopicClassification.labels,
               TopicClassification.classified_at,
               LowConfidenceQuestion.raw_question, LowConfidenceQuestion.source,
               ReviewQueue.normalized_question, ReviewQueue.review_status,
               ReviewQueue.occurrence_count)
        .join(LowConfidenceQuestion,
              TopicClassification.question_id == LowConfidenceQuestion.id)
        .outerjoin(ReviewQueue, LowConfidenceQuestion.matched_review_id == ReviewQueue.id)
    )
    async with db.async_session() as s:
        rows = (await s.execute(stmt)).all()
    items = []
    for qid, labels, classified_at, raw, source, norm, review_status, occ in rows:
        if label not in (labels or []):
            continue
        items.append({"question_id": qid, "labels": labels,
                      "text": norm or raw, "raw": raw,
                      "normalized": norm is not None, "review_status": review_status,
                      "source": source, "occurrence_count": occ,
                      "classified_at": classified_at.isoformat() if classified_at else None})
    total = len(items)
    pages = max(1, -(-total // size))
    page = max(1, page)
    start = (page - 1) * size
    return {"label": label, "total": total, "page": page, "pages": pages, "size": size,
            "items": items[start:start + size]}
