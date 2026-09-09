# app/core/retrieval.py
# ch04:在线与评估复用同一检索函数,strategy 切四策略——「评估的就是线上的」。
from app.config import settings
from app.core import embeddings, rerank
from app.kb import milvus_client


def arrange_head_tail(items: list) -> list:
    """最相关放首、次相关放尾,其余按序居中(缓解 lost-in-the-middle)。"""
    if len(items) <= 2:
        return items
    return [items[0], *items[2:], items[1]]


async def search_knowledge(
    query: str, strategy: str = "hybrid_rerank", top_k: int | None = None,
    category: str | None = None, client=None, collection: str = "knowledge",
) -> list[dict]:
    """strategy: vector | bm25 | hybrid | hybrid_rerank。
    返回最终排序的 hit(hybrid_rerank 含 rerank_score);不做首尾组装(调用侧按需)。"""
    top_k = top_k or settings.rerank_top_k
    client = client or milvus_client.get_client()
    milvus_client.ensure_collection(client, collection=collection)

    if strategy == "vector":
        vec = await embeddings.embed_query(query)
        return milvus_client.dense_search(client, vec, top_k, category, collection)
    if strategy == "bm25":
        return milvus_client.bm25_search(client, query, top_k, category, collection)

    # hybrid / hybrid_rerank:先混合召回 recall_top_k
    vec = await embeddings.embed_query(query)
    hits = milvus_client.hybrid_search(
        client, vec, query, settings.recall_top_k, settings.recall_top_k, category, collection)
    if strategy == "hybrid":
        return hits[:top_k]

    # hybrid_rerank:对召回结果精排
    docs = [f"{h['question']} {h['answer']}" for h in hits]
    ranked = await rerank.rerank(query, docs, top_n=top_k)
    out = []
    for idx, score in ranked:
        h = dict(hits[idx]); h["rerank_score"] = score
        out.append(h)
    return out
