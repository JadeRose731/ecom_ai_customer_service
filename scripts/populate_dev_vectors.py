# scripts/populate_dev_vectors.py
"""dev 专用(无嵌入 key):把 MySQL 里的 chunk 按真实文本 + 确定性占位向量灌进 Milvus。

为什么需要:pytest 的 milvus fixture 每轮 drop/重建集合,跑完回归集合就是空的,
评估/冒烟前要重灌。行构造与 dualwrite.vectorize_pending 完全一致
(text=category+questions+answer,sparse 由集合上的 BM25 Function 从 text 真实生成),
差别只有两点:
  1. dense 用 sha256 确定性占位向量——vector/hybrid 检索路无意义,BM25 完全真实;
  2. **不回填 MySQL status**——pending 保留,真 key 的 kb-vectorize 按 PK=id 原位覆盖。
只处理 status=pending 的块,绝不碰已 done 的块(那边是真向量)。
运行:PYTHONPATH=. uv run python scripts/populate_dev_vectors.py
"""
import asyncio
import hashlib

from app.db import repository
from app.kb import milvus_client


def _placeholder(chunk_id: int) -> list[float]:
    out: list[float] = []
    seed = str(chunk_id).encode()
    while len(out) < milvus_client.DIM:
        seed = hashlib.sha256(seed).digest()
        out.extend((b - 128) / 128.0 for b in seed)
    return out[: milvus_client.DIM]


async def main() -> None:
    chunks = await repository.list_pending_chunks()
    if not chunks:
        print("没有 pending 块。若 MySQL 已 done 而集合为空,直接 make kb-vectorize(幂等重嵌)。")
        return
    client = milvus_client.get_client()
    milvus_client.ensure_collection(client)
    rows = [
        {"id": r.id, "dense": _placeholder(r.id),
         "text": f"{r.category}\n{r.questions}\n{r.answer}",
         "question": r.questions, "answer": r.answer,
         "section_path": r.section_path or "", "content_type": r.content_type or "",
         "category": r.category or ""}
        for r in chunks
    ]
    for i in range(0, len(rows), 64):
        milvus_client.upsert_vectors(client, rows[i:i + 64])
    milvus_client.flush(client)  # Standalone Bounded 一致性:flush 后 BM25 才可检索
    print(f"✅ dev 占位灌库:{len(rows)} 块(status 仍 pending;真 key 后 make kb-vectorize 原位覆盖)。")


if __name__ == "__main__":
    asyncio.run(main())
