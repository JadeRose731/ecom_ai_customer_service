"""ch09 观测与成本只读 API:三块报表(意图成本账/评估趋势/置信度校准)各自报态、互不连坐。
成本与校准读 data/ch09/reports/ 的脚本产物(数在脚本里算好,这里原样透出);
趋势的权威源是 eval_runs 表,页面直接读表,不落中间 json。
产物缺失/写坏半个 json 一律按「没跑过」处理——坏文件不许炸整页。"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter

from app.config import settings
from app.db.repository import list_eval_runs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/observability")

_ROOT = Path(__file__).resolve().parents[2]
_COST_JSON = _ROOT / "data" / "ch09" / "reports" / "cost_by_intent.json"
_CAL_JSON = _ROOT / "data" / "ch09" / "reports" / "confidence_calibration.json"


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cost_block() -> dict:
    data = _load_json(_COST_JSON)
    if not isinstance(data, dict):   # 缺失或形状写坏(顶层非 dict)都按没跑过
        return {"present": False, "job": "cost-report",
                "hint": "还没跑过按意图成本账(Langfuse 在跑时点右侧「重跑」)"}
    rows = data.get("rows") or []
    if not rows:
        return {"present": True, "status": "missing", "rows": [],
                "hint": "窗口内没有带 intent tag 的 trace——先去聊天页聊几句再重跑"}
    return {"present": True, "status": "ok", "days": data.get("days"),
            "rows": rows, "top": rows[0]}


def _cal_block() -> dict:
    data = _load_json(_CAL_JSON)
    if not isinstance(data, dict):
        return {"present": False, "job": "calibrate-confidence",
                "hint": "还没跑过阈值校准(milvus + 上游可用时点「重跑」,校准完回填在用阈值)"}
    current = settings.evidence_confidence_threshold
    rec = data.get("recommended_threshold")
    try:
        in_sync = rec is not None and round(rec, 2) == round(float(current), 2)
    except (TypeError, ValueError):  # 形状写坏(推荐阈值非数值)——按不同步处理,不炸整页
        in_sync = False
    return {"present": True, "recommended_threshold": rec, "youden_j": data.get("youden_j"),
            "current_threshold": current, "in_sync": in_sync,
            "distribution": data.get("distribution") or {},
            "scan": data.get("scan") or []}


async def _trend_block() -> dict:
    try:
        runs = await list_eval_runs(limit=10)
    except Exception as e:  # noqa: BLE001 —— 趋势读挂只挂自己那块
        logger.warning("eval_runs 读取失败: %s", e)
        return {"status": "error", "runs": [], "job": "eval-flywheel",
                "hint": "eval_runs 读不到(检查 mysql)"}
    if not runs:
        return {"status": "absent", "runs": [], "job": "eval-flywheel",
                "hint": "还没跑过评估流水线(上游可用时点「重跑」)"}
    return {"status": "ok", "runs": [
        {"id": r.id, "triggered_by": r.triggered_by,
         "created_at": r.created_at.strftime("%m-%d %H:%M") if r.created_at else None,
         "metrics": r.metrics} for r in runs]}


@router.get("/overview")
async def overview() -> dict:
    return {"cost": _cost_block(), "trend": await _trend_block(), "calibration": _cal_block()}
