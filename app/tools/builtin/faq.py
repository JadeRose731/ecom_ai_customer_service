# app/tools/builtin/faq.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.config import settings
from app.core import query_understanding, retrieval, selfcheck
from app.tools import registry


class FaqInput(BaseModel):
    keyword: str = Field(description="用户咨询的政策/规则/操作类问题(可用原话)")
    category: str | None = Field(default=None, description="可选:按品类过滤,如『运费』『退货』『商品手册』")


@tool(args_schema=FaqInput)
async def query_faq(keyword: str, category: str | None = None) -> dict:
    """查询常见问题/政策知识库(混合检索+重排)。用于政策、规则、时效、费用、商品手册等通用问题。
    返回带编号证据供作答引用;证据不足时返回 sufficient=False,请据此向用户拒答。"""
    # ch04 RAG 管线:①Query 理解 → ②混合召回+重排 → ③机械闸 → ④自评闸 → ⑤首尾组装编号
    u = await query_understanding.understand(keyword)
    query = u["standard"]
    # 同义词扩展只拼进检索文本(检索侧),不改标准问法语义
    search_query = query + (" " + " ".join(u["expanded"]) if u["expanded"] else "")

    hits = await retrieval.search_knowledge(
        search_query, strategy="hybrid_rerank", category=category)

    # 机械闸:无召回 or 最高分低于阈值
    top = hits[0]["rerank_score"] if hits else 0.0
    if not hits or top < settings.rerank_min_score:
        return {"sufficient": False, "source": "retrieval_low_conf",
                "reason": f"检索证据不足(top={top:.3f})", "citations": []}

    # 语义闸:生成前自评证据够不够
    ev_texts = [f"{h['question']} {h['answer']}" for h in hits]
    chk = await selfcheck.check_sufficient(query, ev_texts)
    if not chk["useful"]:
        return {"sufficient": False, "source": "self_check",
                "reason": chk["reason"], "citations": []}

    # 首尾组装 + 编号
    arranged = retrieval.arrange_head_tail(hits)
    citations = [
        {"n": i + 1, "id": h["id"], "section_path": h["section_path"],
         "question": h["question"], "answer": h["answer"], "content_type": h["content_type"]}
        for i, h in enumerate(arranged)
    ]
    evidence = "\n".join(f"[{c['n']}] {c['question']}: {c['answer']}" for c in citations)
    return {"sufficient": True, "evidence": evidence, "citations": citations}


registry.register(registry.spec_from_langchain_tool(query_faq, source="builtin", timeout=30.0))
# query_faq 走 RAG 管线(改写+混合检索+重排+自评,多次上游往返),远慢于默认 5s
