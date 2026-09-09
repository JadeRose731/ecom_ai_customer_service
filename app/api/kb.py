# app/api/kb.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core import jobs, retrieval
from app.db import repository
from app.kb import chunking, dedup, documents, dualwrite, milvus_client, sources

router = APIRouter(prefix="/api/kb")


class PreviewReq(BaseModel):
    content_type: str
    content: str | None = None      # 手工贴的正文
    file: str | None = None         # 或材料清单里的文件名


class IngestReq(BaseModel):
    content_type: str
    content: str
    vectorize: bool = False         # 「入库后顺手向量化」


class SearchReq(BaseModel):
    query: str
    top_k: int | None = None


def _load_markdown(req: PreviewReq) -> str:
    """校验入参并取出正文;非法一律 400。"""
    if req.content_type not in sources.CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"非法 content_type:{req.content_type}")
    if req.file is not None:
        if req.file not in sources.SOURCE_TYPES:
            raise HTTPException(status_code=400, detail=f"文件不在材料清单里:{req.file}")
        return (sources.KB_DIR / req.file).read_text(encoding="utf-8")
    if req.content is None or not req.content.strip():
        raise HTTPException(status_code=400, detail="正文为空")
    return req.content


def _preview_blocks(chunks: list[documents.Chunk], existing_fps: set[str]) -> dict:
    blocks = []
    seen = set(existing_fps)
    n_table = 0
    for c in chunks:
        is_table = bool(chunking.is_table_block(c.answer))
        n_table += is_table
        fp = dedup.fingerprint(c.questions, c.answer)
        duplicate = fp in seen
        seen.add(fp)
        blocks.append({
            "section_path": c.section_path,
            "questions": c.questions,
            "answer": c.answer,
            "chars": len(c.answer),
            "is_key_clause": bool(c.is_key_clause),
            "is_table": is_table,
            "duplicate": duplicate,
        })
    # 切块特性触发:表格按行拆(同表多块)、超长递归切(同节多块且非表格)
    by_q: dict[str, int] = {}
    for b in blocks:
        by_q[b["questions"]] = by_q.get(b["questions"], 0) + 1
    return {
        "total": len(blocks),
        "blocks": blocks,
        "flags": {
            "table_rows_split": n_table >= 2,
            "long_text_split": any(
                n > 1 for q, n in by_q.items() if q) and not all(
                b["is_table"] for b in blocks if b["questions"] and by_q[b["questions"]] > 1),
        },
    }


@router.get("/overview")
async def overview() -> dict:
    # 读不到 ≠ 对不上:任何一路读不到,对应数回 None,consistent 也是 None
    try:
        stats = await repository.knowledge_stats()
    except Exception:
        stats = None
    try:
        client = milvus_client.get_client()
        milvus_count = milvus_client.count(client)
    except Exception:
        milvus_count = None
    if stats is None or milvus_count is None:
        consistent = None
    else:
        consistent = stats["done"] == milvus_count
    try:
        staging = await repository.staging_stats()
    except Exception:
        staging = None
    src = [
        {"file": fname, "content_type": ctype,
         "exists": (sources.KB_DIR / fname).is_file()}
        for fname, ctype in sources.SOURCE_TYPES.items()
    ]
    return {
        "stats": stats,
        "milvus_count": milvus_count,
        "consistent": consistent,
        "staging": staging,
        "sources": src,
        "content_types": list(sources.CONTENT_TYPES),
        "jobs": {
            name: {k: v for k, v in st.items() if k != "log_tail"}
            for name, st in jobs.all_status().items()
        },
    }


@router.post("/preview")
async def preview(req: PreviewReq) -> dict:
    md = _load_markdown(req)
    chunks = documents.build_chunks(md, content_type=req.content_type)
    try:
        pairs = await repository.list_chunk_pairs()
        fps = {dedup.fingerprint(q, a) for q, a in pairs}
    except Exception:
        fps = set()  # 库读不到时预览照常,只是查重列显示不出
    return _preview_blocks(chunks, fps)


@router.post("/ingest")
async def ingest(req: IngestReq) -> dict:
    md = _load_markdown(PreviewReq(content_type=req.content_type, content=req.content))
    chunks = documents.build_chunks(md, content_type=req.content_type)
    if not chunks:
        raise HTTPException(status_code=400, detail="正文切不出任何知识块")
    # 查重指纹 = 问法 + 正文;批内 + 库内都查。同节多块正文不同,不会被误杀。
    pairs = await repository.list_chunk_pairs()
    fps = {dedup.fingerprint(q, a) for q, a in pairs}
    new, skipped = [], 0
    for c in chunks:
        fp = dedup.fingerprint(c.questions, c.answer)
        if fp in fps:
            skipped += 1
            continue
        fps.add(fp)
        new.append(c)
    # 双写顺序不许反:先写 MySQL(pending),再向量化;失败不回滚
    ids = await dualwrite.write_pending(new)
    inserted = len(ids)
    out = {"inserted": inserted, "skipped": skipped, "total_chunks": len(chunks)}
    if req.vectorize and inserted:
        try:
            client = milvus_client.get_client()
            milvus_client.ensure_collection(client)
            out["vectorized"] = await dualwrite.vectorize_pending(client)
            client.flush(milvus_client.COLLECTION)
        except Exception:
            raise HTTPException(
                status_code=502,
                detail=f"已入库 {inserted} 块(pending),向量化失败——补跑一次即可",
            )
    return out


@router.post("/vectorize")
async def vectorize() -> dict:
    try:
        client = milvus_client.get_client()
        milvus_client.ensure_collection(client)
        n = await dualwrite.vectorize_pending(client)
        client.flush(milvus_client.COLLECTION)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"向量化失败:{e}")
    return {"vectorized": n}


@router.post("/search")
async def search(req: SearchReq) -> dict:
    try:
        hits = await retrieval.search_knowledge(req.query, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"检索失败:{e}")
    return {"hits": hits}


@router.get("/staging")
async def staging(status: str | None = None) -> dict:
    try:
        if status:
            rows = await repository.list_staging_by_status(status)
        else:
            rows = (await repository.list_staging_by_status("extracted")
                    + await repository.list_staging_by_status("kept")
                    + await repository.list_staging_by_status("discarded"))
        stats = await repository.staging_stats()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"暂存表读不到:{e}")
    return {
        "stats": stats,
        "items": [{
            "id": r.id, "batch_no": r.batch_no, "source_ref": r.source_ref,
            "question": r.question, "answer": r.answer, "status": r.status,
        } for r in rows],
    }
