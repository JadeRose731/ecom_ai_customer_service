# app/core/rerank.py
# ch04:重排直连上游 /rerank。它不是 OpenAI 协议里的东西,走 Jina / Cohere 那套形状
# (query + documents,回 results 里的 index 与 relevance_score),所以手写请求。
import httpx

from app.config import settings

_RERANK_URL = settings.rerank_base_url.rstrip("/") + "/rerank"


async def _post(url, json, headers, timeout):
    async with httpx.AsyncClient() as c:
        return await c.post(url, json=json, headers=headers, timeout=timeout)


async def rerank(query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
    """调 bge-reranker-v2-m3。返回 [(原始索引, 相关分)] 按分降序,截断 top_n;空 docs 安全返回 []。"""
    if not docs:
        return []
    payload = {"model": settings.rerank_model, "query": query, "documents": docs,
               "top_n": top_n or len(docs), "return_documents": True}
    resp = await _post(_RERANK_URL, payload,
                       {"Authorization": f"Bearer {settings.rerank_api_key}"}, 60)
    resp.raise_for_status()
    results = resp.json()["results"]
    ranked = sorted(((r["index"], float(r["relevance_score"])) for r in results),
                    key=lambda x: x[1], reverse=True)
    return ranked[:top_n] if top_n else ranked
