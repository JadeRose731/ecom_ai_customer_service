"""ch09 观测页只读 API:三块报表各自报态互不连坐——没产物指作业、跑过没数提示先聊、
写坏半个 json 按「没跑过」处理;趋势块权威源是 eval_runs 表,读挂了只挂自己那块。"""
import json

import pytest

from app.api import observability as obs_api
from app.config import settings
from app.db import repository


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """产物路径指到 tmp(默认两文件都不存在 = 从没跑过)。"""
    cost = tmp_path / "cost_by_intent.json"
    cal = tmp_path / "confidence_calibration.json"
    monkeypatch.setattr(obs_api, "_COST_JSON", cost)
    monkeypatch.setattr(obs_api, "_CAL_JSON", cal)
    return {"cost": cost, "cal": cal}


def _write(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


async def test_all_absent_points_to_jobs(client, reports, db_session_factory, db_clean):
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["cost"]["present"] is False
    assert body["cost"]["job"] == "cost-report"
    assert body["calibration"]["present"] is False
    assert body["calibration"]["job"] == "calibrate-confidence"
    assert body["trend"]["status"] == "absent"     # 测试库无轮次
    assert body["trend"]["job"] == "eval-flywheel"


async def test_cost_rows_pass_through_top_is_first(client, reports):
    rows = [
        {"intent": "商品咨询", "count": 3, "tokens": 120, "avg_tokens": 40, "share": 0.6},
        {"intent": "物流", "count": 2, "tokens": 80, "avg_tokens": 40, "share": 0.4},
    ]
    _write(reports["cost"], {"days": 7, "rows": rows})
    r = await client.get("/api/observability/overview")
    cost = r.json()["cost"]
    assert cost["present"] is True and cost["status"] == "ok"
    assert cost["rows"] == rows                    # 逐字段原样透出,页面不算一遍
    assert cost["top"] == rows[0]                  # 最烧钱 = 产物第一行(脚本已降序)


async def test_cost_missing_when_no_tagged_traces(client, reports):
    _write(reports["cost"], {"days": 7, "rows": []})
    r = await client.get("/api/observability/overview")
    cost = r.json()["cost"]
    assert cost["present"] is True and cost["status"] == "missing"
    assert "聊" in cost["hint"]


async def test_trend_reads_eval_runs_newest_first(client, db_session_factory, db_clean):
    await repository.insert_eval_run("手动", 300, {"recall_at_10": 0.9, "mrr": 0.8})
    await repository.insert_eval_run("定时", 300, {"recall_at_10": 0.8, "mrr": 0.9})
    r = await client.get("/api/observability/overview")
    trend = r.json()["trend"]
    assert trend["status"] == "ok"
    assert len(trend["runs"]) == 2
    assert trend["runs"][0]["id"] > trend["runs"][1]["id"]   # 新在上
    assert trend["runs"][0]["metrics"]["recall_at_10"] == 0.8   # 后插的在新
    assert trend["runs"][0]["triggered_by"] == "定时"
    assert trend["runs"][1]["metrics"]["recall_at_10"] == 0.9


async def test_calibration_in_sync_flag(client, reports, monkeypatch):
    _write(reports["cal"], {"recommended_threshold": 0.55, "youden_j": 0.42,
                            "distribution": {"answerable": {"n": 180}, "absent": {"n": 60}},
                            "scan": [{"t": 0.5, "tpr": 0.9, "fpr": 0.2, "j": 0.7}]})
    # 在用 0.5 vs 推荐 0.55 → 不同步
    r = await client.get("/api/observability/overview")
    cal = r.json()["calibration"]
    assert cal["present"] is True
    assert cal["in_sync"] is False
    assert cal["current_threshold"] == settings.evidence_confidence_threshold
    # 在用值回填成推荐值 → 同步
    monkeypatch.setattr(settings, "evidence_confidence_threshold", 0.55)
    cal = (await client.get("/api/observability/overview")).json()["calibration"]
    assert cal["in_sync"] is True


async def test_broken_json_treated_as_absent(client, reports):
    reports["cost"].write_text('{"days": 7, "rows": [', encoding="utf-8")   # 半个 json
    reports["cal"].write_text("not json at all", encoding="utf-8")
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200                    # 不 500
    body = r.json()
    assert body["cost"]["present"] is False
    assert body["calibration"]["present"] is False


async def test_shape_corrupt_json_does_not_crash(client, reports):
    # ch09 review #5:形状写坏(顶层非 dict / 推荐阈值非数值)同款不许炸——只挂自己那块
    reports["cost"].write_text("[1, 2, 3]", encoding="utf-8")   # 合法 json 但顶层是数组
    reports["cal"].write_text(json.dumps(
        {"recommended_threshold": "很高", "youden_j": None, "distribution": {}, "scan": []},
        ensure_ascii=False), encoding="utf-8")
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["cost"]["present"] is False                    # 非 dict → 按没跑过
    assert body["calibration"]["present"] is True
    assert body["calibration"]["in_sync"] is False             # 非数值推荐 → 判不同步,不抛
    assert body["trend"]["status"] == "absent"                 # 其余块照常


async def test_trend_error_does_not_hurt_others(client, reports, monkeypatch):
    _write(reports["cost"], {"days": 7, "rows": [
        {"intent": "闲聊", "count": 1, "tokens": 9, "avg_tokens": 9, "share": 1.0}]})
    _write(reports["cal"], {"recommended_threshold": 0.5, "youden_j": 0.4,
                            "distribution": {}, "scan": []})
    async def boom(limit=10):
        raise RuntimeError("mysql 挂了")
    monkeypatch.setattr(obs_api, "list_eval_runs", boom)
    r = await client.get("/api/observability/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["trend"]["status"] == "error"
    assert body["cost"]["present"] is True         # 两外块照样出
    assert body["calibration"]["in_sync"] is True
