"""CLI(kb-reset job,heavy):清空知识库两表 + drop Milvus 集合。
重建源就是 data/kb 材料文档与历史对话,清掉可重灌。运行:PYTHONPATH=. uv run python scripts/reset_kb.py"""
import asyncio

from sqlalchemy import text

from app.db.base import engine


async def main() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
        await conn.execute(text("TRUNCATE TABLE knowledge_chunks"))
        await conn.execute(text("TRUNCATE TABLE qa_extraction_staging"))
        await conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    print("  MySQL:knowledge_chunks / qa_extraction_staging 已清空")

    from app.kb import milvus_client
    client = milvus_client.get_client()
    if client.has_collection(milvus_client.COLLECTION):
        client.drop_collection(milvus_client.COLLECTION)
        print("  Milvus:集合 knowledge 已 drop")
    else:
        print("  Milvus:集合本就不存在")
    print("✅ 知识库已重置。下一步:make kb-build && make kb-vectorize")


if __name__ == "__main__":
    asyncio.run(main())
