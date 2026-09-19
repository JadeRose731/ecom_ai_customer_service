"""ch10 阈值扫描重演:验证集经 :8110 打分一次,分数表固定,九条候选线各算一遍 micro-F1。
consistent 记「重演当选线 == threshold.json 在用线」——证明线上阈值可以从数据复算,
不是手拍的一个数。口径与 train.py 一比一:裸 scores >= t(无兜底)、严格大于留任先到的线
(打平时 0.45 压住 0.50)。运行:make ch10-threshold-scan(需分类器服务 :8110)。"""
import json
import pathlib
import sys
from datetime import datetime

import httpx

VAL = pathlib.Path("data/ch10/dataset/val.jsonl")
ONNX_DIR = pathlib.Path("data/ch10/onnx")       # 服务在用的 threshold.json 在这儿
MODEL_DIR = pathlib.Path("data/ch10/model")     # 训练侧原件,兜底读
REPORTS = pathlib.Path("data/ch10/reports")
SERVICE = "http://127.0.0.1:8110"
CANDIDATES = [round(0.30 + 0.05 * i, 2) for i in range(9)]   # 0.30 ~ 0.70,同 train.py


def _micro_f1(preds: list[set[str]], gold: list[set[str]]) -> tuple[float, int, int, int]:
    tp = sum(len(p & g) for p, g in zip(preds, gold))
    fp = sum(len(p - g) for p, g in zip(preds, gold))
    fn = sum(len(g - p) for p, g in zip(preds, gold))
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom else 0.0     # 全零也按 0 分算,同 sklearn zero_division=0
    return f1, tp, fp, fn


async def main() -> int:
    val = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    scores_table: list[dict[str, float]] = []
    async with httpx.AsyncClient(timeout=300) as client:
        for i in range(0, len(val), 64):        # 打分一次,分数表固定
            r = await client.post(f"{SERVICE}/classify",
                                  json={"texts": [s["text"] for s in val[i:i + 64]]})
            r.raise_for_status()
            scores_table += [res["scores"] for res in r.json()["results"]]
    gold = [set(s["labels"]) for s in val]

    scan = []
    best_t, best_f1 = None, -1.0
    for t in CANDIDATES:
        preds = [{lb for lb, sc in row.items() if sc >= t} for row in scores_table]
        f1, tp, fp, fn = _micro_f1(preds, gold)
        scan.append({"threshold": t, "micro_f1": round(f1, 6), "tp": tp, "fp": fp, "fn": fn})
        if f1 > best_f1:                        # 严格大于:打平先到的留任
            best_t, best_f1 = t, f1

    src = ONNX_DIR / "threshold.json"
    if not src.exists():
        src = MODEL_DIR / "threshold.json"
    in_use = json.loads(src.read_text())["threshold"]
    consistent = abs(best_t - in_use) < 1e-9

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "threshold_scan.json").write_text(json.dumps(
        {"ran_at": datetime.now().isoformat(timespec="seconds"),
         "val_size": len(val), "scan": scan,
         "best_threshold": best_t, "best_micro_f1": round(best_f1, 6),
         "in_use_threshold": in_use, "consistent": consistent},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"重演最优线 {best_t}(micro-F1 {best_f1:.4f}),在用线 {in_use}(read from {src});"
          f"{'一致 ✓ 阈值可复算' if consistent else '不一致 ✗ 阈值对不上,查训练扫描口径'};"
          f"报告已落 {REPORTS}/threshold_scan.json")
    return 0 if consistent else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK:✓/✗ 不炸
        sys.stdout.reconfigure(errors="replace")
    sys.exit(__import__("asyncio").run(main()))
