"""ch09 置信度阈值校准:拿 ch04 评估集,A/B/C 桶(应可答)与 D 桶(应拒答)分别算
evidence_confidence 分布,扫阈值取 Youden J(=可答通过率 - 应拒放行率)最大的分离点。
输出分布表 + 推荐阈值 → 人工回填 settings.evidence_confidence_threshold(不拍脑袋)。
运行:make calibrate-confidence(需 milvus + 上游可用 + 知识库已建)。
产物:dev-notes/ch09-confidence-calibration.txt
"""
import asyncio
import json
import pathlib
import sys

from app.core import retrieval
from app.core.confidence import compute_evidence_confidence

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-confidence-calibration.txt"
ANSWERABLE = {"A_policy", "B_model", "C_colloquial"}
_SEM = asyncio.Semaphore(8)
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


async def _conf(sample) -> tuple[str, float]:
    async with _SEM:
        hits = await retrieval.search_knowledge(sample["query"], strategy="hybrid_rerank")
    return sample["bucket"], compute_evidence_confidence(hits).score


def _dist(name, xs):
    xs = sorted(xs)
    if not xs:
        return
    p = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]
    _log(f"{name:12s} n={len(xs):3d} min={xs[0]:.3f} p25={p(.25):.3f} "
         f"p50={p(.5):.3f} p75={p(.75):.3f} max={xs[-1]:.3f}")


async def main():
    samples = [json.loads(ln) for ln in open(_ROOT / "tests/data/eval_ch04.jsonl", encoding="utf-8")]
    results = await asyncio.gather(*(_conf(s) for s in samples))
    answerable = [c for b, c in results if b in ANSWERABLE]
    absent = [c for b, c in results if b == "D_absent"]

    _log("=== evidence_confidence 分布(hybrid_rerank)===")
    _dist("可答(ABC)", answerable)
    _dist("应拒(D)", absent)

    _log("\n=== 阈值扫描(通过率=conf>=t 的可答占比;放行率=conf>=t 的应拒占比)===")
    _log(f"{'t':>6s} {'可答通过率':>10s} {'应拒放行率':>10s} {'YoudenJ':>8s}")
    best_t, best_j = 0.0, -1.0
    for i in range(5, 96):
        t = i / 100
        tpr = sum(1 for c in answerable if c >= t) / len(answerable)
        fpr = sum(1 for c in absent if c >= t) / len(absent)
        j = tpr - fpr
        if i % 5 == 0 or j > best_j:
            _log(f"{t:6.2f} {tpr:10.3f} {fpr:10.3f} {j:8.3f}")
        if j > best_j:
            best_t, best_j = t, j
    _log(f"\n推荐阈值 evidence_confidence_threshold = {best_t:.2f}(Youden J={best_j:.3f})")
    _log("请回填 app/config.py 默认值,并在 dev-notes/ch09.md 记录本次校准。")
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
