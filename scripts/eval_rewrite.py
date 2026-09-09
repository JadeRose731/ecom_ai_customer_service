"""Query 理解标注验证(需真实聊天上游):口语问法 → standard+expanded 应含期望关键词。"""
import asyncio
import json

from app.core.query_understanding import understand


async def main() -> None:
    ok = tot = 0
    with open("tests/data/query_rewrite_samples.jsonl", encoding="utf-8") as f:
        for ln in f:
            s = json.loads(ln)
            tot += 1
            r = await understand(s["query"])
            blob = r["standard"] + " " + " ".join(r["expanded"])
            hit = any(w in blob for w in s["expect_any"])
            print("OK" if hit else "MISS", s["query"], "->", r)
            ok += hit
    print(f"{ok}/{tot}")
    assert ok >= tot * 4 / 5, "口语归一命中率应 ≥4/5"


if __name__ == "__main__":
    asyncio.run(main())
