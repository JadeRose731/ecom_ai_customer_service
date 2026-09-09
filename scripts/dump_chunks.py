"""临时工具:打印全部源文档的切块结构(section_path / questions / answer 摘要),供评估集校准。"""
from app.kb import documents, sources


def main() -> None:
    total = 0
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks = documents.build_chunks(md, content_type=ctype)
        print(f"===== {fname} ({ctype}) : {len(chunks)} 块")
        for c in chunks:
            answer = c.answer.replace("\n", "⏎")
            print(f"[{c.section_path}] Q={c.questions} key={c.is_key_clause}\n    A={answer}")
        total += len(chunks)
    print(f"TOTAL {total}")


if __name__ == "__main__":
    main()
