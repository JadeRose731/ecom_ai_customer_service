# 环境适配:计划用临时 Milvus Lite 文件,Windows 无 milvus-lite 包;
# 经用户确认改跑 Docker Milvus Standalone(见 dev-notes/ch03),夹具逐用例 drop/重建集合。
# 用非单例客户端(uri 传参),避免进程内 _ensured 缓存吞掉 drop 后的重建。
import pytest

from app.config import settings
from app.kb import milvus_client as mc


@pytest.fixture()
def client():
    c = mc.get_client(uri=settings.milvus_uri)
    if c.has_collection(mc.COLLECTION):
        c.drop_collection(mc.COLLECTION)
    mc.ensure_collection(c)
    yield c
    if c.has_collection(mc.COLLECTION):
        c.drop_collection(mc.COLLECTION)


def _vec(seed: float) -> list[float]:
    # 造 1024 维、方向可区分的向量
    v = [0.0] * mc.DIM
    v[0] = seed
    v[1] = 1.0 - seed
    return v


def _row(cid: int, seed: float, q: str, a: str) -> dict:
    # ch04 新 schema:dense 向量 + BM25 text 必填(稀疏向量由 BM25 Function 自动产出)
    return {"id": cid, "dense": _vec(seed), "text": f"{q} {a}", "question": q, "answer": a,
            "section_path": "p", "content_type": "faq", "category": "c"}


def test_upsert_then_search_returns_fields(client):
    mc.upsert_vectors(client, [
        _row(1, 1.0, "运费怎么算", "满99包邮"),
        _row(2, 0.0, "发货时效", "48小时内发货"),
    ])
    # Standalone 默认 Bounded 一致性,写后立查有 ~1s 可见性延迟;flush 落盘即立即可见
    client.flush(mc.COLLECTION)
    hits = mc.search(client, _vec(0.98), top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == 1
    assert hits[0]["question"] == "运费怎么算"
    assert hits[0]["answer"] == "满99包邮"
    assert isinstance(hits[0]["score"], float)


def test_upsert_is_idempotent_by_pk(client):
    row = _row(1, 1.0, "q", "a")
    mc.upsert_vectors(client, [row])
    mc.upsert_vectors(client, [row])  # 同 id 重写
    client.flush(mc.COLLECTION)
    assert mc.count(client) == 1
