# scripts/judge_check.py
"""裁判一致性回归:台账里已处置且有证据快照的个案,用快照原样重放——不重检索、不重生成,
只换裁判。人工处置翻译成标准答案:已解决=确属编造(该判 false),无需解决=误判(该判 true),
未解决跳过。输出逐条对照与一致率;不一致退出码 1。
上游偶发回空 completion(结构化解析成 None)时重试一次,仍失败记「调用失败」不计入一致。
运行:PYTHONPATH=. uv run python scripts/judge_check.py"""
import asyncio
import sys

from pydantic import BaseModel

from app.core.llm import get_chat_model
from app.core.prompts import FAITHFULNESS_PROMPT
from app.db import repository

_PAGE = 100


class _Faith(BaseModel):
    faithful: bool


def expected_verdict(status: str) -> bool | None:
    """处置 → 标准答案:已解决=该判编造(False);无需解决=该判忠实(True);其余不考。"""
    return {"resolved": False, "wontfix": True}.get(status)


def eligible(case: dict) -> bool:
    """能考的个案:已处置 且 带 citations 快照(没快照就等于要重检索,那不是回归)。"""
    return expected_verdict(case["status"]) is not None and bool(case.get("citations"))


def build_evidence(citations: list[dict]) -> str:
    """快照拼回 evidence 文本:与线上 _evidence_text 逐字一致;老快照缺 n 按序补。"""
    lines = [f"[{c.get('n') or i + 1}] {c.get('question', '')}: {c.get('answer', '')}"
             for i, c in enumerate(citations)]
    return "\n".join(lines) or "(无证据)"


def tally(rows: list[tuple[str, bool | None, bool | None]]) -> dict:
    """(eval_id, 期望, 裁判) → 对照账。调用失败(None)不计入一致/不一致,单列。"""
    agree = disagree = failed = 0
    misses: list[str] = []
    for eid, expected, judged in rows:
        if judged is None:
            failed += 1
        elif judged == expected:
            agree += 1
        else:
            disagree += 1
            misses.append(eid)
    graded = agree + disagree
    return {"agree": agree, "disagree": disagree, "failed": failed,
            "graded": graded, "rate": round(agree / graded, 4) if graded else None,
            "misses": misses}


async def _judge(chain, case: dict) -> bool | None:
    evidence = build_evidence(case["citations"])
    for _ in range(2):  # 偶发空 completion / 抖动:重试一次
        try:
            r = await chain.ainvoke({"evidence": evidence, "answer": case["answer"]})
        except Exception as e:  # noqa: BLE001
            print(f"  [judge] {case['eval_id']} 调用异常: {type(e).__name__}")
            continue
        if r is not None:
            return r.faithful
        print(f"  [judge] {case['eval_id']} 空 completion,重试")
    return None


async def main() -> int:
    cases: list[dict] = []
    page = 1
    while True:
        d = await repository.list_faith_cases(None, page, _PAGE)
        cases.extend(d["items"])
        if page * _PAGE >= d["total"]:
            break
        page += 1
    exam = [c for c in cases if eligible(c)]
    print(f"# 裁判一致性回归  台账 {len(cases)} 条,可考 {len(exam)} 条"
          f"(已处置且有证据快照;未解决/无快照跳过)")
    if not exam:
        print("没有可考个案:先跑 make eval-rag 产出编造个案并人工处置。")
        return 0

    chain = FAITHFULNESS_PROMPT | get_chat_model(temperature=0).with_structured_output(_Faith)
    rows: list[tuple[str, bool | None, bool | None]] = []
    for c in exam:
        expected = expected_verdict(c["status"])
        judged = await _judge(chain, c)
        label = "调用失败(重试后仍空),不计入一致" if judged is None else (
            "✅" if judged == expected else "❌")
        want = "该判编造" if expected is False else "该判忠实"
        got = "— " if judged is None else ("裁判=编造" if judged is False else "裁判=忠实")
        print(f"  {c['eval_id']:<8} ({c['status']} → {want}): {got} {label}")
        rows.append((c["eval_id"], expected, judged))

    t = tally(rows)
    print(f"一致率: {t['agree']}/{t['graded']}"
          + (f" = {t['rate']:.2%}" if t["rate"] is not None else "")
          + (f"  调用失败 {t['failed']} 条" if t["failed"] else "")
          + (f"  不一致: {','.join(t['misses'])}" if t["misses"] else ""))
    return 1 if t["disagree"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
