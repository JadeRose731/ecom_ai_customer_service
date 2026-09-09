# scripts/validate_eval_ch04.py
"""评估集守门:五桶 300 题,四件事不过就 exit 1。

  1. id / query 全局不重复,五桶题数一致(各 60);
  2. 每组 expect_section(或 E 桶的每组别名)至少命中一个真实 chunk 的 section_path;
  3. expect_points 逐字(去空白后)出现在目标小节正文里——A–D 看 expect_section 匹配的块,
     E 看任一组别名匹配的块;
  4. D 桶不带 ground truth(空证据 + should_refuse);A/B/C/E 必须带且不拒答。

语料从 data/kb/*.md 现场切块(与入库同一条 build_chunks 路径),不需要 MySQL/Milvus 在场。
"""
import sys
from collections import Counter
from pathlib import Path

from app.kb import documents, sources

_ROOT = Path(__file__).resolve().parent.parent
_EVAL_PATH = _ROOT / "tests" / "data" / "eval_ch04.jsonl"

GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial", "E_multi"]
ALL_BUCKETS = GRADED_BUCKETS + ["D_absent"]
PER_BUCKET = 60


def _norm(s: str) -> str:
    return "".join((s or "").split())


def _load_corpus() -> list:
    chunks = []
    for fname, ctype in sources.SOURCE_TYPES.items():
        md = (sources.KB_DIR / fname).read_text(encoding="utf-8")
        chunks.extend(documents.build_chunks(md, content_type=ctype))
    return chunks


def main() -> int:
    import json

    rows = [json.loads(l) for l in _EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    corpus = _load_corpus()
    problems: list[str] = []

    def bad(msg: str) -> None:
        problems.append(msg)

    # 1. 唯一性 + 五桶题数一致
    seen_id: dict[str, str] = {}
    seen_q: dict[str, str] = {}
    bucket_counts = Counter(r.get("bucket") for r in rows)
    for r in rows:
        if r.get("id") in seen_id:
            bad(f"{r.get('id')}: id 重复(先出现在 {seen_id[r['id']]})")
        seen_id.setdefault(r.get("id"), r.get("id"))
        if r.get("query") in seen_q:
            bad(f"{r.get('id')}: query 重复(与 {seen_q[r['query']]} 同题)")
        seen_q.setdefault(r.get("query"), r.get("id"))
    for b in ALL_BUCKETS:
        if bucket_counts[b] != PER_BUCKET:
            bad(f"桶 {b}: {bucket_counts[b]} 题,应为 {PER_BUCKET}")
    stray = set(bucket_counts) - set(ALL_BUCKETS)
    if stray:
        bad(f"未知桶: {sorted(stray)}")

    # 2+3+4. 逐题校验
    for r in rows:
        rid, b = r.get("id"), r.get("bucket")
        if b not in ALL_BUCKETS:
            continue
        if b == "E_multi":
            groups = r.get("expect_sections_all") or []
            if not groups or not all(isinstance(g, list) and g for g in groups):
                bad(f"{rid}: E 桶 expect_sections_all 必须是非空的组列表")
                continue
            if "expect_section" in r:
                bad(f"{rid}: E 桶不应带 expect_section")
        else:
            groups = [r.get("expect_section") or []]
        points = r.get("expect_points") or []
        refuse = r.get("should_refuse")

        if b == "D_absent":
            if any(groups) or points or not refuse:
                bad(f"{rid}: D 桶必须空证据 + should_refuse=true")
            continue
        if refuse is not False:
            bad(f"{rid}: {b} 桶应可作答(should_refuse=false)")
        if not points:
            bad(f"{rid}: {b} 桶缺少 expect_points")
            continue

        # 2. 每组别名至少命中一个真实小节
        group_chunks: list[list] = []
        for gi, aliases in enumerate(groups, 1):
            matched = [c for c in corpus
                       if any(_norm(a) in _norm(c.section_path) for a in aliases)]
            if not matched:
                bad(f"{rid}: 第 {gi} 组 {aliases} 在语料中无匹配小节")
            group_chunks.append(matched)

        # 3. 要点逐字落在目标小节正文里
        pool = [c for matched in group_chunks for c in matched]
        for p in points:
            if not any(_norm(p) in _norm(c.answer) for c in pool):
                where = "任一组的匹配小节" if b == "E_multi" else str(groups[0])
                bad(f"{rid}: 要点「{p}」不在 {where} 的正文里")

    if problems:
        print(f"评估集校验失败:{len(problems)} 处")
        for p in problems:
            print(f"  - {p}")
        return 1
    n_groups = sum(len(r.get("expect_sections_all", [r.get("expect_section", [])]))
                   for r in rows if r["bucket"] != "D_absent")
    print(f"评估集校验通过:{len(rows)} 题(五桶各 {PER_BUCKET}),"
          f"语料 {len(corpus)} 块,证据组 {n_groups} 组,要点 "
          f"{sum(len(r.get('expect_points', [])) for r in rows)} 条全部逐字命中。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
