"""ch09 自动化评估流水线:复用 ch04 评估集与指标,定期跑、落 eval_runs、连趋势。
指标:检索段 Recall@10 / MRR(hybrid_rerank 线上策略,可答桶),生成段 Faithfulness + D 桶拒答率。
运行:make eval-flywheel(TRIGGER=手动|定时,默认手动)。cron 示例:
  0 6 * * * cd /path/to/mewhelp && make eval-flywheel TRIGGER=定时 >> log/eval.log 2>&1
趋势:读最近 10 轮,对比上一轮涨跌;任何指标下滑标 ⚠——README:「重点全在这条趋势线上」。
产物:dev-notes/ch09-eval-trend.txt

实现注(偏差⑥,dev-notes 有全记录):Plan import 的 _format_evidence/_hit_rank/_retrieve/
_refusal_one/_mean 在 eval_ch04 里并不存在,按现树真实形状复用:_evidence_text / _group_rank
(E 桶多组语义)/ retrieval.search_knowledge / query_faq(D 桶线上管线含两道闸)+ 本地 _mean;
_Faith 裁判模型与并发/超时常量直接 import 复用,不复制粘贴。
"""
import asyncio
import pathlib
import sys

from app.core import retrieval
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.db import repository
from app.tools.builtin.faq import query_faq
from scripts.eval_ch04 import (
    CALL_TIMEOUT, GEN_CONCURRENCY, GRADED_BUCKETS, K, RETR_CONCURRENCY,
    _Faith, _bucket_sections, _evidence_text, _group_rank, _load, _try,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-eval-trend.txt"
_LINES: list[str] = []


def _log(msg=""):
    print(msg, flush=True)
    _LINES.append(msg)


def _mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


async def _retrieve_one(sem, s):
    async with sem:
        hits = await _try(
            retrieval.search_knowledge(s["query"], strategy="hybrid_rerank", top_k=K),
            f"retrieval:{s['id']}")
        return s, hits or []


async def _gen_faith(sem, s, hits, model, judge):
    async with sem:
        evidence = _evidence_text(hits)
        ans = await _try((RAG_ANSWER_PROMPT | model).ainvoke(
            {"query": s["query"], "evidence": evidence}), f"gen:{s['id']}")
        if ans is None:
            return None
        text = ans.content if isinstance(ans.content, str) else str(ans.content)
        fa = await _try((FAITHFULNESS_PROMPT | judge).ainvoke(
            {"evidence": evidence, "answer": text}), f"faith:{s['id']}")
        return None if fa is None else bool(fa.faithful)


async def _refusal_one(sem, s):
    async with sem:
        r = await _try(query_faq(s["query"]), f"refuse:{s['id']}")
        return bool(r and r.get("sufficient") is False)


async def run_once() -> dict:
    samples = _load()
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_absent"]

    retr_sem = asyncio.Semaphore(RETR_CONCURRENCY)
    gen_sem = asyncio.Semaphore(GEN_CONCURRENCY)

    # 检索段:Recall@10 / MRR(分组语义同 ch04:A–D 单组,Recall 看前 K 名、MRR 全窗口;
    # E 跨文档桶按组给分——Recall=命中组数占比,MRR=各组名次倒数取均值)
    hits_list = await asyncio.gather(*(_retrieve_one(retr_sem, s) for s in graded))
    rs, rrs = [], []
    for s, hits in hits_list:
        groups = _bucket_sections(s)
        if s["bucket"] == "E_multi":
            ranks = [_group_rank(hits, g) for g in groups]
            rs.append(sum(1 for r in ranks if r) / len(ranks))
            rrs.append(sum(1.0 / r if r else 0.0 for r in ranks) / len(ranks))
        else:
            recall_rank = _group_rank(hits, groups[0], K)
            mrr_rank = _group_rank(hits, groups[0])
            rs.append(1.0 if recall_rank else 0.0)
            rrs.append(1.0 / mrr_rank if mrr_rank else 0.0)
    recall, mrr = _mean(rs), _mean(rrs)

    # 生成段:答案→裁判判忠实;D 桶走线上 query_faq 端到端(含置信度闸,拒答=sufficient 为 False)
    model = get_chat_model()
    judge = get_chat_model(temperature=0).with_structured_output(_Faith)   # 裁判去抖:0 才能当尺子(承 ch04)
    faiths = await asyncio.gather(*(
        _gen_faith(gen_sem, s, h, model, judge) for s, h in hits_list))
    faithfulness = _mean([1.0 if f else 0.0 for f in faiths if f is not None])

    refusals = await asyncio.gather(*(_refusal_one(gen_sem, s) for s in absent))
    refusal_rate = _mean([1.0 if r else 0.0 for r in refusals])

    return {"dataset_size": len(samples),
            "metrics": {"recall_at_10": round(recall, 3), "mrr": round(mrr, 3),
                        "faithfulness": round(faithfulness, 3),
                        "refusal_rate": round(refusal_rate, 3)}}


def _trend(runs) -> None:
    """runs 按新→旧;打印每轮各指标,并与上一轮(更旧一行)对比涨跌,下滑标 ⚠。"""
    _log("\n=== 评估趋势(新在上)===")
    names = ["recall_at_10", "mrr", "faithfulness", "refusal_rate"]
    _log(f"{'轮次':>4s} {'时间':16s} {'触发':4s} " + "".join(f"{n:>16s}" for n in names))
    for i, r in enumerate(runs):
        prev = runs[i + 1].metrics if i + 1 < len(runs) else None
        row = f"#{r.id:>3d} {r.created_at:%m-%d %H:%M}   {r.triggered_by:4s} "
        for n in names:
            v = r.metrics.get(n)
            cell = f"{v:.3f}" if v is not None else "—"
            if prev and v is not None and prev.get(n) is not None:
                d = v - prev[n]
                cell += " ↑" if d > 0.005 else (" ⚠↓" if d < -0.005 else " →")
            row += f"{cell:>16s}"
        _log(row)
    latest, older = runs[0].metrics, (runs[1].metrics if len(runs) > 1 else None)
    if older:
        drops = [n for n in names if latest.get(n, 0) < older.get(n, 0) - 0.005]
        _log(f"\n⚠ 下滑指标:{', '.join(drops)}" if drops else "\n所有指标持平或上涨。")


async def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # Windows GBK 控制台:↑⚠→ 打不出也不炸(文件仍 utf-8)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggered-by", default="手动", choices=["手动", "定时"])
    args = ap.parse_args()

    _log(f"=== ch09 评估流水线(触发:{args.triggered_by})===")
    result = await run_once()
    m = result["metrics"]
    _log(f"本轮:Recall@10={m['recall_at_10']:.3f} MRR={m['mrr']:.3f} "
         f"Faithfulness={m['faithfulness']:.3f} 拒答率={m['refusal_rate']:.3f}")
    await repository.insert_eval_run(args.triggered_by, result["dataset_size"], m)
    _trend(await repository.list_eval_runs(limit=10))
    _OUT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _log(f"\n趋势报告已落 {_OUT.relative_to(_ROOT)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
