# app/api/rageval.py
"""RAG 评估报告 API:只读 `data/ch04/reports/rag_eval.json`,一个指标都不在 web 层重算——
页面上的数与终端 `make eval-rag` 跑出来的是同一份。唯一派生量 best(总体 MRR 最高那一路)
在这里算一次,KPI 与读图句都指它。产物缺失/写坏都是「没跑过」状态,不是错误。"""
import json
from pathlib import Path

from fastapi import APIRouter

from app.core import jobs

router = APIRouter(prefix="/api/rag-eval")

REPORT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ch04" / "reports" / "rag_eval.json"


def load_report() -> dict | None:
    """读产物;缺失或半个 json 都回 None(调用方按「没跑过」呈现)。"""
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def derive_best(report: dict) -> str | None:
    """总体 MRR 最高那一路(结论性派生量,只算这一次)。"""
    retrieval = report.get("retrieval") or {}
    best, best_mrr = None, None
    for st, per in retrieval.items():
        mrr = (per or {}).get("all", {}).get("mrr")
        if mrr is None:
            continue
        if best_mrr is None or mrr > best_mrr:
            best, best_mrr = st, mrr
    return best


@router.get("/overview")
async def overview() -> dict:
    report = load_report()
    if report is None:
        return {
            "present": False, "meta": None, "retrieval": None,
            "evidence_coverage": None, "generation": None, "generation_done": False,
            "best": None, "job": jobs.status("eval-rag"),
        }
    return {
        "present": True,
        "meta": report.get("meta"),
        "retrieval": report.get("retrieval"),
        "evidence_coverage": report.get("evidence_coverage"),
        "generation": report.get("generation"),
        "generation_done": bool((report.get("meta") or {}).get("generation_done")),
        "best": derive_best(report),
        "job": jobs.status("eval-rag"),
    }
