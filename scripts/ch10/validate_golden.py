"""ch10 预标质量闸:黄金样例上标签集合完全一致率 ≥ 80% 才放行批量预标。
运行:make ch10-golden(需上游可用)。不过线就改 prompt,不改黄金样例来凑分。
报告一式两份:控制台给人看,golden_report.json 给验收页读(不过线也要落,页面才看得出闸是红的)。"""
import asyncio
import json
import pathlib
import sys
from datetime import datetime

from scripts.ch10.prelabel import prelabel_batch

GOLDEN = pathlib.Path(__file__).parent / "golden_samples.jsonl"
PASS_RATE = 0.8
REPORTS = pathlib.Path("data/ch10/reports")


async def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK:✗ 等字符不炸
        sys.stdout.reconfigure(errors="replace")
    samples = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    predicted = await prelabel_batch([s["text"] for s in samples])
    hits = 0
    failures = []
    for s, pred in zip(samples, predicted):
        ok = set(pred) == set(s["labels"])
        hits += ok
        if ok:
            continue
        failures.append({"text": s["text"], "gold": list(s["labels"]), "pred": list(pred)})
        print(f"✗ {s['text']}\n    标准: {s['labels']}  预标: {pred}")
    rate = hits / len(samples)
    passed = rate >= PASS_RATE
    # 诊断:预标异常一律兜底成「其他」,全兜底 = 多半上游不可用,别误导人去改 prompt
    upstream_note = None
    if not passed and failures and all(f["pred"] == ["其他"] for f in failures):
        upstream_note = ("全部预标都落到兜底「其他」——多半是上游不可用而非 prompt 问题,"
                         "先查 CHAT_* 连通(网关 502/超时)再回来重跑")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "golden_report.json").write_text(json.dumps(
        {"ran_at": datetime.now().isoformat(timespec="seconds"),
         "total": len(samples), "hits": hits, "rate": round(rate, 4),
         "pass_line": PASS_RATE, "passed": passed, "upstream_note": upstream_note,
         "failures": failures},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n黄金样例 {len(samples)} 条,集合全对 {hits} 条,通过率 {rate:.0%}(闸线 {PASS_RATE:.0%})"
          f";报告已落 {REPORTS}/golden_report.json")
    if upstream_note:
        print(f"⚠ {upstream_note}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
