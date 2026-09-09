"""CLI(kb-preview job):材料清单 + 就地切块预览,打印各文件块数与切块特性触发。
运行:PYTHONPATH=. uv run python scripts/preview_kb.py"""
from app.kb import chunking, documents, sources


def main() -> None:
    total = 0
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        n_table = sum(bool(chunking.is_table_block(c.answer)) for c in chunks)
        n_key = sum(c.is_key_clause for c in chunks)
        total += len(chunks)
        print(f"  {fname} [{ctype}]:{len(chunks)} 块(关键条款 {n_key},表格块 {n_table})")
    print(f"✅ 预览:共 {total} 块。与 /kb ② 区页面上看到的应当一致。")


if __name__ == "__main__":
    main()
