# 环境适配:计划用临时 Milvus Lite 文件,Windows 无 milvus-lite 包;
# 经用户确认改跑 Docker Milvus Standalone(见 dev-notes/ch03),夹具逐用例 drop/重建集合。
import pytest

from app.kb import milvus_client as mc


@pytest.fixture()
def client():
    c = mc.get_client()
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


def test_upsert_then_search_returns_fields(client):
    mc.upsert_vectors(client, [
        {"id": 1, "vector": _vec(1.0), "question": "运费怎么算", "answer": "满99包邮"},
        {"id": 2, "vector": _vec(0.0), "question": "发货时效", "answer": "48小时内发货"},
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
    row = {"id": 1, "vector": _vec(1.0), "question": "q", "answer": "a"}
    mc.upsert_vectors(client, [row])
    mc.upsert_vectors(client, [row])  # 同 id 重写
    client.flush(mc.COLLECTION)
    assert mc.count(client) == 1
