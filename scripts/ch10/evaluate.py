"""ch10 评测:留出测试集上每类 P/R/F1 + 每类混淆矩阵 + 容错红线 + 判错样本导出。
运行:make ch10-eval。评测集扎在自家电商场景(dataset/test.jsonl),不引公开榜单。"""
import json
import pathlib
import sys

import numpy as np
import torch
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.taxonomy import LABEL2ID, NUM_CLASSES, SEVERITY, TOPIC_NAMES

MODEL_DIR = pathlib.Path("data/ch10/model")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
RED_LINE = 0.8   # 严档类目 F1 红线


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

    REPORTS.mkdir(parents=True, exist_ok=True)
    lines = ["# ch10 分类器评测报告(留出测试集)", "",
             f"测试集 {len(samples)} 条;判定阈值 {threshold}(验证集扫描所得)。", "",
             f"**micro**: P={micro_p:.3f} R={micro_r:.3f} F1={micro_f1:.3f}  |  "
             f"**macro**: P={macro_p:.3f} R={macro_r:.3f} F1={macro_f1:.3f}", "",
             "## 每类指标(容错档位:严档 F1 红线 0.8)", "",
             "| 类目 | 档位 | P | R | F1 | support | 红线 |",
             "|---|---|---|---|---|---|---|"]
    for i, name in enumerate(TOPIC_NAMES):
        sev = SEVERITY[name]
        flag = "🔴 不达标,先回头搞数据" if sev == "严" and f1[i] < RED_LINE else "✅"
        lines.append(f"| {name} | {sev} | {p[i]:.3f} | {r[i]:.3f} | {f1[i]:.3f} "
                     f"| {int(support[i])} | {flag} |")
    lines += ["", "## 每类混淆矩阵(TN FP / FN TP)", ""]
    for i, name in enumerate(TOPIC_NAMES):
        tn, fp = cms[i][0]
        fn, tp = cms[i][1]
        lines.append(f"- **{name}**: TN={tn} FP={fp} FN={fn} TP={tp}")
    (REPORTS / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")

    err = ["# ch10 判错样本(人工复核:错在哪一类?标注本身有没有毛病?)", ""]
    for i, s in enumerate(samples):
        pred_labels = [TOPIC_NAMES[j] for j in range(NUM_CLASSES) if preds[i][j]]
        if set(pred_labels) != set(s["labels"]):
            err.append(f"- {s['text']}\n  标准: {s['labels']}  预测: {pred_labels}")
    (REPORTS / "error_samples.md").write_text("\n".join(err), encoding="utf-8")
    print(f"micro-F1 {micro_f1:.4f} / macro-F1 {macro_f1:.4f};"
          f"报告与判错样本已落 {REPORTS}/")


if __name__ == "__main__":
    main()
