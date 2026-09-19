"""ch10 ONNX 导出:torch 模型 → onnx(变长 batch/seq),导出后跑测试集校验与 torch 预测完全一致。
报告一式两份:控制台给人看,export_report.json 给验收页读——校验不过也先落报告再退出,
页面才看得出「上次导出没过」,而不是一片空白。
注意:torch 2.9+ 的 torch.onnx.export 默认 dynamo=True,导 HF 模型此处显式 dynamo=False
走 TorchScript 导出器 + dynamic_axes 稳定路线(Context7 查证)。运行:make ch10-export。"""
import json
import pathlib
import shutil
import sys
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = pathlib.Path("data/ch10/model")
OUT = pathlib.Path("data/ch10/onnx")
TEST = pathlib.Path("data/ch10/dataset/test.jsonl")
REPORTS = pathlib.Path("data/ch10/reports")
OPSET = 17


class LogitsWrapper(torch.nn.Module):
    """HF 模型输出是 ModelOutput dict,导出前包一层只回 logits 张量。
    必须 eval:torch 2.14 TorchScript 导出器按根模块 training 标志走 HF SDPA 分支,
    不 eval 会把 train 态的注意力 dropout 随机烘进图里,导出产物每次都不一样。"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.eval()

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):   # Windows 控制台 GBK 兜底
        sys.stdout.reconfigure(errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    wrapper = LogitsWrapper(model)
    sample = tokenizer(["买大了想退", "快递到哪了"], padding=True, return_tensors="pt")
    dyn = {0: "batch", 1: "seq"}
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(OUT / "model.onnx"),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": dyn, "attention_mask": dyn,
                      "token_type_ids": dyn, "logits": {0: "batch"}},
        dynamo=False,
        opset_version=OPSET,
    )
    tokenizer.save_pretrained(OUT)          # 带出 tokenizer.json 给轻运行时用
    shutil.copy(MODEL_DIR / "threshold.json", OUT / "threshold.json")

    # 一致性校验:测试集全量,按行计账——logits 超容差与过线标签翻面都算不一致
    threshold = json.loads((OUT / "threshold.json").read_text())["threshold"]
    texts = [json.loads(l)["text"]
             for l in TEST.read_text(encoding="utf-8").splitlines() if l.strip()]
    sess = ort.InferenceSession(str(OUT / "model.onnx"),
                                providers=["CPUExecutionProvider"])
    mismatch = 0
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = tokenizer(texts[i:i + 32], truncation=True, padding=True,
                            max_length=128, return_tensors="pt")
            t_logits = wrapper(**enc).numpy()
            (o_logits,) = sess.run(["logits"], {k: v.numpy() for k, v in enc.items()})
            close = np.isclose(t_logits, o_logits, atol=1e-3).all(axis=1)
            t_pred = (1 / (1 + np.exp(-t_logits)) >= threshold)
            o_pred = (1 / (1 + np.exp(-o_logits)) >= threshold)
            mismatch += int((~close | (t_pred != o_pred).any(axis=1)).sum())

    report = {"ran_at": datetime.now().isoformat(timespec="seconds"),
              "checked": len(texts), "mismatch": mismatch, "passed": mismatch == 0,
              "onnx_path": str(OUT / "model.onnx"),
              "onnx_bytes": (OUT / "model.onnx").stat().st_size, "opset": OPSET}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if mismatch:
        raise SystemExit(f"ONNX 与 torch 预测不一致 {mismatch} 条,导出失败"
                         f"(报告已落 {REPORTS}/export_report.json)")
    print(f"ONNX 导出并校验通过({len(texts)} 条预测完全一致):{OUT}/model.onnx;"
          f"报告已落 {REPORTS}/export_report.json")


if __name__ == "__main__":
    main()
