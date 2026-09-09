# app/api/rageval.py
"""RAG 评估报告 API:只读 `data/ch04/reports/rag_eval.json`,一个指标都不在 web 层重算——
页面上的数与终端 `make eval-rag` 跑出来的是同一份。唯一派生量 best(总体 MRR 最高那一路)
在这里算一次,KPI 与读图句都指它。产物缺失/写坏都是「没跑过」状态,不是错误。

编造个案台账是本路由下唯一的写入口:分页查台账 + 处置状态流转(说明必填校验在这里)。"""
import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.core import jobs
from app.db import repository

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
        "hallucination": report.get("hallucination"),
        "generation_done": bool((report.get("meta") or {}).get("generation_done")),
        "best": derive_best(report),
        "job": jobs.status("eval-rag"),
    }


@router.get("/faith-cases")
async def list_faith_cases(
    status: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
) -> dict:
    if status is not None and status not in ("unresolved", "resolved", "wontfix"):
        raise HTTPException(status_code=422, detail="status 仅限 unresolved/resolved/wontfix")
    return await repository.list_faith_cases(status, page, size)


class _CaseStatusBody(BaseModel):
    status: Literal["unresolved", "resolved", "wontfix"]
    resolution: str = ""


@router.post("/faith-cases/{case_id}/status")
async def set_faith_case_status(case_id: int, body: _CaseStatusBody) -> dict:
    """处置流转。已解决/无需解决必须带说明(空白也算没填);退回未解决不需要,说明一并清空。"""
    if body.status != "unresolved" and not body.resolution.strip():
        raise HTTPException(status_code=400, detail="处置必须写说明")
    ok = await repository.set_faith_case_status(
        case_id, body.status, body.resolution.strip() or None)
    if not ok:
        raise HTTPException(status_code=404, detail="个案不存在")
    return {"ok": True, "status": body.status}
