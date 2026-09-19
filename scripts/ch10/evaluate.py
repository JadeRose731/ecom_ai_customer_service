"""ch10 评测:留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。
报告一式两份:eval_report.md 给人读,eval_report.json 给验收页读(同一份数据两种渲)。
运行:make ch10-eval。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。"""
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
RED_LINES = {"严": 0.8, "中": 0.6}   # 宽档不设线,页面画 —(中档线 Plan 未钉,取 0.6 记 dev-notes)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def predict(model, tokenizer, texts: list[str], threshold: float, device: str) -> np.ndarray:
    model.eval()
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt").to(device)
            probs_all.append(torch.sigmoid(model(**enc).logits).cpu().numpy())
    probs = np.concatenate(probs_all)
    preds = (probs >= threshold).astype(int)
    for r in range(len(preds)):                    # 全不过线取最高分兜底,不产出空标签
        if preds[r].sum() == 0:
            preds[r][probs[r].argmax()] = 1
    return preds


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK 兜底
        sys.stdout.reconfigure(errors="replace")
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    threshold = json.loads((MODEL_DIR / "threshold.json").read_text())["threshold"]
    samples = [json.loads(l) for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    texts = [s["text"] for s in samples]
    gold = np.zeros((len(samples), NUM_CLASSES), dtype=int)
    for i, s in enumerate(samples):
        for lb in s["labels"]:
            gold[i][LABEL2ID[lb]] = 1
    preds = predict(model, tokenizer, texts, threshold, device)

    p, r, f1, support = precision_recall_fscore_support(gold, preds, zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="micro", zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        gold, preds, average="macro", zero_division=0)
    cms = multilabel_confusion_matrix(gold, preds)

    # 结构化指标先算齐,.md/.json 各渲一份(口径铁律:同一份数据,两种读法)
    classes = []
    for i, name in enumerate(TOPIC_NAMES):
        sev = SEVERITY[name]
        line = RED_LINES.get(sev)
        classes.append({"name": name, "severity": sev,
                        "p": round(float(p[i]), 4), "r": round(float(r[i]), 4),
                        "f1": round(float(f1[i]), 4), "support": int(support[i]),
                        "red_line": line,
                        "passed": None if line is None else bool(f1[i] >= line),
                        "tn": int(cms[i][0][0]), "fp": int(cms[i][0][1]),
                        "fn": int(cms[i][1][0]), "tp": int(cms[i][1][1])})
    errors = []
    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        if set(pred_labels) == set(s["labels"]):
            continue
        missed = [lb for lb in s["labels"] if lb not in pred_labels]
        extra = [lb for lb in pred_labels if lb not in s["labels"]]
        kind = "错位" if missed and extra else ("漏打" if missed else "多打")
        errors.append({"text": s["text"], "gold": list(s["labels"]), "pred": pred_labels,
                       "missed": missed, "extra": extra, "kind": kind,
                       "matrix_entries": len(missed) + len(extra)})
    red_line_passed = all(c["passed"] is not False for c in classes)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {"ran_at": datetime.now().isoformat(timespec="seconds"),
              "test_size": len(samples), "threshold": threshold,
              "red_lines": RED_LINES,
              "micro": {"p": round(float(micro_p), 4), "r": round(float(micro_r), 4),
                        "f1": round(float(micro_f1), 4)},
              "macro": {"p": round(float(macro_p), 4), "r": round(float(macro_r), 4),
                        "f1": round(float(macro_f1), 4)},
              "classes": classes, "errors": errors,
              "total_cells": sum(e["matrix_entries"] for e in errors),
              "total_fp": sum(len(e["extra"]) for e in errors),
              "total_fn": sum(len(e["missed"]) for e in errors),
              "red_line_passed": red_line_passed}
    (REPORTS / "eval_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# ch10 分类器评测报告(留出测试集)", "",
             f"测试集 {len(samples)} 条;判定阈值 {threshold}(验证集扫描所得)。", "",
             f"**micro**: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}", "",
             "## 每类指标(容错红线:严档 0.8 / 中档 0.6,宽档不设线)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for c in classes:
        if c["passed"] is None:
            flag = "—"
        elif c["passed"]:
            flag = "✅"
        else:
            flag = f"🔴 不达标(F1<{c['red_line']}),先回头搞数据"
        lines.append(f"| {c['name']} | {c['severity']} | {c['p']:.3f} | {c['r']:.3f} | {c['f1']:.3f} "
                     f"| {c['support']} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    for c in classes:
        lines.append(f"- **{c['name']}**: TN={c['tn']} FP={c['fp']} FN={c['fn']} TP={c['tp']}")
    (REPORTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    err = ["# ch10 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    for e in errors:
        err.append(f"- [{e['kind']}] {e['text']}\n"
                   f"  标准: {e['gold']}  预测: {e['pred']}(漏 {e['missed']} / 多 {e['extra']})")
    (REPORTS / "error_samples.md").write_text("\n".join(err), encoding="utf-8")
    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f};"
          f"报告(.md/.json)与判错样本已落 {REPORTS}/")


if __name__ == "__main__":
    main()
