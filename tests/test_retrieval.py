# ch03 起留的真实 Milvus 检索测试。ch04 起 search_knowledge 走 strategy 分支:
# 阈值过滤移到 query_faq 的机械闸(rerank_min_score,见 test_query_faq_rag),
# 这里只验 vector 路打真库能取回正确 chunk。
import pytest

from app.config import settings
from app.core import retrieval
from app.kb import milvus_client


@pytest.fixture()
def milvus():
    c = milvus_client.get_client(uri=settings.milvus_uri)
    if c.has_collection(milvus_client.COLLECTION):
        c.drop_collection(milvus_client.COLLECTION)
    milvus_client.ensure_collection(c)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "dense": v_hit, "text": "运费怎么算 满99包邮", "question": "运费怎么算", "answer": "满99包邮",
         "section_path": "p", "content_type": "faq", "category": "c"},
        {"id": 2, "dense": v_other, "text": "发货时效 48小时", "question": "发货时效", "answer": "48小时",
         "section_path": "p", "content_type": "faq", "category": "c"},
    ])
    c.flush(milvus_client.COLLECTION)
    yield c
    if c.has_collection(milvus_client.COLLECTION):
        c.drop_collection(milvus_client.COLLECTION)


async def test_vector_strategy_returns_top_hit(monkeypatch, milvus):
    query_vec = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)

    async def fake_embed(q):
        return query_vec

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed)
    hits = await retrieval.search_knowledge("邮费是多少", strategy="vector", top_k=1, client=milvus)
    assert hits and hits[0]["question"] == "运费怎么算"
    assert hits[0]["score"] > 0.99  # Standalone COSINE distance 即相似度,同向应接近 1
