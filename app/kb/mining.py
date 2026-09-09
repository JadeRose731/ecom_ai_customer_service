# app/kb/mining.py
from pydantic import BaseModel, Field

from app.core.llm import get_chat_model
from app.core.prompts import MINING_PROMPT
from app.db import repository
from app.kb import dedup, dualwrite
from app.kb.documents import Chunk


class QaPair(BaseModel):
    question: str = Field(description="通用问法,去掉具体订单号/人名")
    answer: str = Field(description="客服原答,不编造")


class QaExtraction(BaseModel):
    pairs: list[QaPair] = Field(default_factory=list, description="抽出的问答对,可为空")


async def extract_qa(conversation_texts: list[str], model=None) -> list[QaPair]:
    model = model or get_chat_model()
    chain = MINING_PROMPT | model.with_structured_output(QaExtraction)
    result: QaExtraction = await chain.ainvoke({"conversations": "\n---\n".join(conversation_texts)})
    return result.pairs


async def _load_conversation_texts() -> list[tuple[str, str]]:
    """返回 [(source_ref, 对话文本)];对话文本由该会话的 user/assistant 消息拼成。"""
    convs = await repository.list_conversations_with_messages()
    out = []
    for conv_id, msgs in convs:
        lines = [f"{m.role}: {m.content}" for m in msgs if m.content]
        if lines:
            out.append((f"conv:{conv_id}", "\n".join(lines)))
    return out


async def mine(batch_size: int = 20, model=None) -> dict:
    sources = await _load_conversation_texts()
    batch_no = f"mine-{len(sources)}"  # 以来源数标批,重跑可覆盖语义
    # 1) 分批抽取 → 写 staging(extracted)
    for start in range(0, len(sources), batch_size):
        batch = sources[start:start + batch_size]
        pairs = await extract_qa([t for _, t in batch], model=model)
        for p in pairs:
            await repository.insert_staging(batch_no, batch[0][0], p.question, p.answer)
    # 2) 整体去重(staging extracted 内 + 对已有 knowledge)
    staged = await repository.list_staging_by_status("extracted")
    existing = await repository.list_all_questions()
    kept, discarded = dedup.dedupe(staged, existing)
    await repository.set_staging_status([s.id for s in kept], "kept")
    await repository.set_staging_status([s.id for s in discarded], "discarded")
    # 3) kept 写 knowledge_chunks(pending, content_type=mined, questions=真实问法)
    chunks = [Chunk(category="历史对话", questions=s.question, answer=s.answer,
                    section_path="mined", content_type="mined") for s in kept]
    if chunks:
        await dualwrite.write_pending(chunks)
    return {"sources": len(sources), "extracted": len(staged),
            "kept": len(kept), "discarded": len(discarded)}
