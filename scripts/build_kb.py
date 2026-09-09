"""CLI:把 data/kb/*.md 切块写入 knowledge_chunks(pending)。之后跑 vectorize_kb.py。
运行:PYTHONPATH=. uv run python scripts/build_kb.py"""
import asyncio

from app.kb import documents, dualwrite, sources


async def main() -> None:
    total = 0
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        ids = await dualwrite.write_pending(chunks)
        total += len(ids)
        print(f"  {fname}: {len(ids)} 块")
    print(f"✅ 建库(pending):共 {total} 块。下一步:scripts/vectorize_kb.py")


if __name__ == "__main__":
    asyncio.run(main())
