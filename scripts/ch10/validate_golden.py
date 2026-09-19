"""ch10 预标质量闸:黄金样例上标签集合完全一致率 ≥ 80% 才放行批量预标。
运行:make ch10-golden(需上游可用)。不过线就改 prompt,不改黄金样例来凑分。"""
import asyncio
import json
import pathlib
import sys

from scripts.ch10.prelabel import prelabel_batch

GOLDEN = pathlib.Path(__file__).parent / "golden_samples.jsonl"
PASS_RATE = 0.8


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK:✗ 等字符不炸
        sys.stdout.reconfigure(errors="replace")
    samples = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    predicted = await prelabel_batch([s["text"] for s in samples])
    hits = 0
    for s, pred in zip(samples, predicted):
        ok = set(pred) == set(s["labels"])
        hits += ok
        if not ok:
            print(f"✗ {s['text']}\n    标准: {s['labels']}  预标: {pred}")
    rate = hits / len(samples)
    print(f"\n黄金样例 {len(samples)} 条,集合全对 {hits} 条,通过率 {rate:.0%}(闸线 {PASS_RATE:.0%})")
    return 0 if rate >= PASS_RATE else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
