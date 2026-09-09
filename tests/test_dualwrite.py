import pytest

from app.db import repository
from app.kb import dualwrite, milvus_client
from app.kb.documents import Chunk


def _chunk(q, a):
    return Chunk(category="c", questions=q, answer=a, section_path="c / " + q, content_type="policy")


# 计划的 milvus 夹具用临时 Lite 文件;Windows 无 Lite,改用 conftest 的 Docker Standalone 夹具。
# 计划片段没带 db_clean,count 断言会吃跨文件残留;补上。


async def test_write_pending_links_neighbors(db_session_factory, db_clean):
    ids = await dualwrite.write_pending([_chunk("q1", "a1"), _chunk("q2", "a2"), _chunk("q3", "a3")])
    assert len(ids) == 3
    by_id = {c.id: c for c in await repository.list_pending_chunks()}
    assert by_id[ids[1]].prev_chunk_id == ids[0]
    assert by_id[ids[1]].next_chunk_id == ids[2]
    assert by_id[ids[0]].prev_chunk_id is None


async def test_vectorize_resumes_after_crash(db_session_factory, milvus, monkeypatch, db_clean):
    await dualwrite.write_pending([_chunk(f"q{i}", f"a{i}") for i in range(4)])

    # 第一趟:embed 在第 2 批抛错(batch_size=2 → 前 2 条 done,后 2 条仍 pending)
    real_calls = {"n": 0}

    async def flaky_embed(texts):
        real_calls["n"] += 1
        if real_calls["n"] == 2:
            raise RuntimeError("嵌入服务中断")
        return [[float(i), 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for i, _ in enumerate(texts)]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", flaky_embed)
    with pytest.raises(RuntimeError):
        await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert await repository.count_chunks_by_status("done") == 2
    assert await repository.count_chunks_by_status("pending") == 2

    # 第二趟:恢复正常 → 只捡剩下 2 条 pending,补齐;Milvus 无重(按 id upsert)
    async def ok_embed(texts):
        return [[9.0, 0.0, 1.0] + [0.0] * (milvus_client.DIM - 3) for _ in texts]

    monkeypatch.setattr("app.kb.dualwrite.embeddings.embed_texts", ok_embed)
    done_now = await dualwrite.vectorize_pending(milvus, batch_size=2)
    assert done_now == 2
    assert await repository.count_chunks_by_status("pending") == 0
    assert await repository.count_chunks_by_status("done") == 4
    milvus.flush(milvus_client.COLLECTION)  # Standalone Bounded 一致性,立查前先 flush
    assert milvus_client.count(milvus) == 4  # 无重无漏
