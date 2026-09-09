# 计划夹具用临时 Milvus Lite 文件;Windows 无 Lite,改用 conftest 的 Docker Standalone 夹具
import pytest

from app.core import retrieval
from app.kb import milvus_client


@pytest.fixture()
def milvus():
    c = milvus_client.get_client()
    if c.has_collection(milvus_client.COLLECTION):
        c.drop_collection(milvus_client.COLLECTION)
    milvus_client.ensure_collection(c)
    v_hit = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    v_other = [0.0, 1.0, 0.0] + [0.0] * (milvus_client.DIM - 3)
    milvus_client.upsert_vectors(c, [
        {"id": 1, "vector": v_hit, "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": v_other, "question": "发货时效", "answer": "48小时"},
    ])
    c.flush(milvus_client.COLLECTION)
    yield c
    if c.has_collection(milvus_client.COLLECTION):
        c.drop_collection(milvus_client.COLLECTION)


async def test_search_returns_top_hits_above_threshold(monkeypatch, milvus):
    query_vec = [1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro(query_vec))
    hits = await retrieval.search_knowledge("邮费是多少", top_k=1, min_score=0.5, client=milvus)
    assert hits and hits[0]["question"] == "运费怎么算"


async def test_below_threshold_filtered(monkeypatch, milvus):
    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query",
                        lambda q: _coro([1.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3)))
    hits = await retrieval.search_knowledge("邮费", top_k=2, min_score=0.999999, client=milvus)
    assert hits == [] or all(h["score"] >= 0.999999 for h in hits)


async def _coro(v):
    return v
