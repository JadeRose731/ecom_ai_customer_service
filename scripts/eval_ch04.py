# scripts/eval_ch04.py
"""ch04 四策略分桶评估:评估的就是线上的——四个策略全部走 app.core.retrieval.search_knowledge。

三段一次跑齐:
  段1 检索(确定性):四策略 × A/B/C 桶 Recall@K / MRR(按 expect_section 命中 section_path)。
  段2 证据覆盖度(确定性):四策略召回 Top-K 证据机械匹配盖住 expect_points 的比例。
  段3 生成(glm):四策略答案覆盖度(证据→生成→glm 判盖住几个要点)+ hybrid_rerank 忠实度 + D 桶拒答率。

产物: data/ch04/reports/rag_eval.txt(人读日志)+ rag_eval.json(评估页读)。
生成段每个调用带超时、gather(return_exceptions=True) 单点故障隔离;上游不可用时
generation=null 照样落盘,页面标注未完成,恢复后重跑补全。
"""
import asyncio
import json
import time
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import settings
from app.core import retrieval
from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT, RAG_ANSWER_PROMPT
from app.tools.business import query_faq

STRATEGIES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
GRADED_BUCKETS = ["A_policy", "B_model", "C_colloquial"]
K = 10
RETR_CONCURRENCY, GEN_CONCURRENCY, CALL_TIMEOUT = 8, 3, 45.0

# 要点核对裁判:只回一个数(部分覆盖算盖住,数字/同义表述都算)——扁平字段,glm 嵌套 502 的教训
_COVER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是要点核对员。给定用户问题、客服回答和要点清单,判断回答盖住了几条要点。"
               "部分覆盖算盖住;数字、同义表述都算。只回盖住的条数。"),
    ("human", "用户问题:{query}\n\n客服回答:\n{answer}\n\n要点清单:\n{points}"),
])


class _Cover(BaseModel):
    covered_count: int


class _Faith(BaseModel):
    faithful: bool


_ROOT = Path(__file__).resolve().parent.parent
_EVAL_PATH = _ROOT / "tests" / "data" / "eval_ch04.jsonl"
_OUT_DIR = _ROOT / "data" / "ch04" / "reports"
_OUT_TXT = _OUT_DIR / "rag_eval.txt"
_OUT_JSON = _OUT_DIR / "rag_eval.json"

_LINES: list[str] = []          # txt 报告的行缓冲
_ERRS: list[str] = []           # 生成段错误记账(截断入报告)


def _norm(s: str) -> str:
    return "".join((s or "").split())   # 去空白,兼容"每 2 周"↔"每2周"


def _log(line: str = "") -> None:
    print(line)
    _LINES.append(line)


def _err(label: str, e: Exception) -> None:
    if len(_ERRS) < 100:
        _ERRS.append(f"{label}: {type(e).__name__}")


async def _try(coro, label: str):
    """带超时跑单个 glm 调用;超时/异常记账后返回 None,不拖垮整轮。"""
    try:
        return await asyncio.wait_for(coro, CALL_TIMEOUT)
    except Exception as e:  # noqa: BLE001 —— 评估脚本要的是隔离,不是崩溃
        _err(label, e)
        return None


def _load() -> list[dict]:
    rows = [json.loads(l) for l in _EVAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    _log(f"评估集: {len(rows)} 题 "
         + " ".join(f"{b}={sum(1 for r in rows if r['bucket'] == b)}" for b in GRADED_BUCKETS + ["D_absent"]))
    return rows


def _hit_section(hit: dict, sections: list[str]) -> bool:
    sp = _norm(hit.get("section_path", ""))
    return any(_norm(s) in sp for s in sections)


def _coverage_mech(points: list[str], hits: list[dict]) -> float | None:
    """证据覆盖度:expect_points 去空白后是否为召回证据答案文本的子串。"""
    if not points:
        return None
    ev = _norm("\n".join(h.get("answer", "") for h in hits))
    return sum(1 for p in points if _norm(p) in ev) / len(points)


def _evidence_text(hits: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] {h.get('question', '')}: {h.get('answer', '')}"
                     for i, h in enumerate(hits)) or "(无证据)"


async def _deterministic(samples: list[dict]):
    """段1+2:每(策略,题)检索一次,复用同一份 HITS 算 Recall/MRR + 证据覆盖度。"""
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    sem = asyncio.Semaphore(RETR_CONCURRENCY)

    async def _one(strategy: str, s: dict):
        async with sem:
            hits = await _try(retrieval.search_knowledge(s["query"], strategy=strategy, top_k=K),
                              f"retrieval {strategy}/{s['id']}")
            return strategy, s["id"], hits or []

    jobs = [_one(st, s) for st in STRATEGIES for s in graded]
    results = await asyncio.gather(*jobs, return_exceptions=True)
    HITS: dict[tuple[str, str], list] = {}
    for r in results:
        if isinstance(r, Exception):
            _err("retrieval", r)
            continue
        HITS[(r[0], r[1])] = r[2]

    # 段1:Recall@K / MRR(逐桶 + 总体;各桶题数相同,总体取桶均值)
    retrieval_d: dict[str, dict] = {}
    for st in STRATEGIES:
        per_bucket = {}
        for b in GRADED_BUCKETS:
            rs, rr = [], []
            for s in graded:
                if s["bucket"] != b:
                    continue
                hits = HITS.get((st, s["id"]), [])
                rank = next((i + 1 for i, h in enumerate(hits) if _hit_section(h, s["expect_section"])), 0)
                rs.append(1.0 if rank else 0.0)
                rr.append(1.0 / rank if rank else 0.0)
            per_bucket[b] = {"recall": round(sum(rs) / len(rs), 4), "mrr": round(sum(rr) / len(rr), 4)}
        per_bucket["all"] = {
            "recall": round(sum(per_bucket[b]["recall"] for b in GRADED_BUCKETS) / len(GRADED_BUCKETS), 4),
            "mrr": round(sum(per_bucket[b]["mrr"] for b in GRADED_BUCKETS) / len(GRADED_BUCKETS), 4)}
        retrieval_d[st] = per_bucket
        _log(f"[检索 {st}] " + "  ".join(
            f"{b}: R@{K}={per_bucket[b]['recall']:.3f} MRR={per_bucket[b]['mrr']:.3f}"
            for b in GRADED_BUCKETS + ["all"]))

    # 段2:证据覆盖度(确定性,四策略对比)
    coverage_d: dict[str, float] = {}
    for st in STRATEGIES:
        vals = [_coverage_mech(s["expect_points"], HITS.get((st, s["id"]), [])) for s in graded]
        vals = [v for v in vals if v is not None]
        coverage_d[st] = round(sum(vals) / len(vals), 4) if vals else 0.0
    _log("[证据覆盖度] " + "  ".join(f"{st}={coverage_d[st]:.3f}" for st in STRATEGIES))
    return retrieval_d, coverage_d, HITS


async def _generation(samples: list[dict], HITS: dict) -> dict:
    """段3:四策略答案覆盖度 + hybrid_rerank 忠实度 + D 桶拒答率(全部 glm,可整体失败)。"""
    graded = [s for s in samples if s["bucket"] in GRADED_BUCKETS]
    absent = [s for s in samples if s["bucket"] == "D_absent"]
    llm = get_chat_model()
    answer_chain = RAG_ANSWER_PROMPT | llm
    cover_chain = _COVER_PROMPT | llm.with_structured_output(_Cover)
    faith_chain = FAITHFULNESS_PROMPT | llm.with_structured_output(_Faith)
    sem = asyncio.Semaphore(GEN_CONCURRENCY)

    async def _gen_one(strategy: str, s: dict):
        async with sem:
            ev = _evidence_text(HITS.get((strategy, s["id"]), []))
            ans = await _try(answer_chain.ainvoke({"query": s["query"], "evidence": ev}),
                             f"answer {strategy}/{s['id']}")
            if ans is None:
                return strategy, s["id"], None
            cover = await _try(cover_chain.ainvoke({
                "query": s["query"], "answer": ans.content,
                "points": "\n".join(f"{i + 1}. {p}" for i, p in enumerate(s["expect_points"]))}),
                f"cover {strategy}/{s['id']}")
            ratio = None
            if cover is not None:
                ratio = max(0, min(cover.covered_count, len(s["expect_points"]))) / len(s["expect_points"])
            return strategy, s["id"], {"answer": ans.content, "coverage": ratio}

    async def _faith_one(sid: str, answer: str):
        async with sem:
            ev = _evidence_text(HITS.get(("hybrid_rerank", sid), []))
            f = await _try(faith_chain.ainvoke({"evidence": ev, "answer": answer}), f"faith {sid}")
            return sid, None if f is None else f.faithful

    async def _refuse_one(s: dict):
        async with sem:
            r = await _try(query_faq(s["query"]), f"refuse {s['id']}")
            return s["id"], bool(r and r.get("sufficient") is False)

    _log(f"[生成段] 覆盖度 {len(STRATEGIES)}×{len(graded)} + 忠实度 {len(graded)} + 拒答 {len(absent)},"
         f"并发={GEN_CONCURRENCY} 超时={CALL_TIMEOUT:.0f}s")

    gen_res = await asyncio.gather(*[_gen_one(st, s) for st in STRATEGIES for s in graded],
                                   return_exceptions=True)
    # 忠实度复用 hybrid_rerank 段已生成的答案,不重复调生成
    hr_answers = [(r[1], r[2]["answer"]) for r in gen_res
                  if not isinstance(r, Exception) and r[0] == "hybrid_rerank" and r[2]]
    faith_res = await asyncio.gather(*[_faith_one(sid, a) for sid, a in hr_answers],
                                     return_exceptions=True)
    refuse_res = await asyncio.gather(*[_refuse_one(s) for s in absent], return_exceptions=True)
    for group in (gen_res, faith_res, refuse_res):
        for r in group:
            if isinstance(r, Exception):
                _err("generation", r)

    # 答案覆盖度:逐策略逐桶 + 总体
    per = {st: {b: [] for b in GRADED_BUCKETS} for st in STRATEGIES}
    for r in gen_res:
        if isinstance(r, Exception) or not r[2] or r[2]["coverage"] is None:
            continue
        st, sid, payload = r
        b = next(s["bucket"] for s in graded if s["id"] == sid)
        per[st][b].append(payload["coverage"])
    answer_coverage = {}
    for st in STRATEGIES:
        per_bucket = {b: (round(sum(v) / len(v), 4) if v else None) for b, v in per[st].items()}
        allv = [v for b in GRADED_BUCKETS for v in per[st][b]]
        per_bucket["all"] = round(sum(allv) / len(allv), 4) if allv else None
        answer_coverage[st] = per_bucket
        _log(f"[答案覆盖度 {st}] " + "  ".join(
            f"{b}={per_bucket[b]}" for b in GRADED_BUCKETS + ["all"]))

    # 忠实度:hybrid_rerank(线上策略)
    faithful_flags = [x[1] for x in faith_res if not isinstance(x, Exception) and x[1] is not None]
    faith_d = {"rate": round(sum(faithful_flags) / len(faithful_flags), 4) if faithful_flags else None,
               "n": len(faithful_flags)}
    _log(f"[忠实度 hybrid_rerank] rate={faith_d['rate']} n={faith_d['n']}")

    # D 桶拒答率(线上管线端到端,含两道闸)
    refuse_flags = {x[0]: x[1] for x in refuse_res if not isinstance(x, Exception)}
    refused = sum(1 for v in refuse_flags.values() if v)
    detail = [{"id": s["id"], "refused": refuse_flags.get(s["id"], False)} for s in absent]
    refusal_d = {"rate": round(refused / len(absent), 4) if absent else None,
                 "refused": refused, "total": len(absent), "detail": detail}
    _log(f"[D 桶拒答率] {refused}/{len(absent)} = {refusal_d['rate']}")

    return {"answer_coverage": answer_coverage, "faithfulness": faith_d,
            "refusal": refusal_d, "llm_error_count": len(_ERRS),
            "llm_errors": _ERRS[:20]}


def _write_report(retrieval_d: dict, coverage_d: dict, generation_d: dict | None, meta: dict) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"meta": meta, "retrieval": retrieval_d,
              "evidence_coverage": coverage_d, "generation": generation_d}
    _OUT_TXT.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    _OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"报告已写入 {_OUT_TXT.name} / {_OUT_JSON.name}")


async def main() -> None:
    t0 = time.time()
    _log(f"# ch04 RAG 四策略评估  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    samples = _load()
    retrieval_d, coverage_d, HITS = await _deterministic(samples)   # 确定性,始终产出
    generation_d = None
    try:
        generation_d = await _generation(samples, HITS)             # glm,失败不拖垮整轮
    except Exception as e:  # noqa: BLE001
        _log(f"[生成段未完成] {type(e).__name__}: {e}")
    meta = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "elapsed_s": round(time.time() - t0, 1),
            "k": K, "strategies": STRATEGIES, "n_samples": len(samples),
            "models": {"chat": settings.chat_model, "embed": settings.embed_model,
                       "rerank": settings.rerank_model},
            "generation_done": generation_d is not None}
    _write_report(retrieval_d, coverage_d, generation_d, meta)


if __name__ == "__main__":
    asyncio.run(main())
