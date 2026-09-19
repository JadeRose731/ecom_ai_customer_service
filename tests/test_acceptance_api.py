"""ch10 验收 API:产物缺失时全链路 200 + missing 态不喊失败(「没跑过」≠「没过线」)。"""
import httpx
import pytest
from fastapi import FastAPI

from app.api import acceptance as acceptance_api
from app.core import jobs as jobs_core
from app.db import repository


@pytest.fixture
async def client(tmp_path, monkeypatch):
    # 产物目录全部指到空 tmp:所有闸走 missing 态,不依赖本机 data/ch10 现状
    monkeypatch.setattr(acceptance_api, "DATA_DIR", tmp_path)
    monkeypatch.setattr(acceptance_api, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(acceptance_api, "DATASET", tmp_path / "dataset")
    monkeypatch.setattr(acceptance_api, "MODEL_DIR", tmp_path / "model")
    monkeypatch.setattr(acceptance_api, "ONNX", tmp_path / "onnx")

    async def _no_db():
        return {"total": 0, "latest": None, "classes": []}

    monkeypatch.setattr(repository, "topic_distribution", _no_db)

    async def _svc_down():
        return {"online": False}

    monkeypatch.setattr(acceptance_api, "_probe_service", _svc_down)   # :8110 在不在线不该影响测试
    app = FastAPI()
    app.include_router(acceptance_api.router)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


FAIL_WORDS = ("失败", "未达标", "不过线", "不达标")


async def test_overview_missing_products_still_200(client):
    r = await client.get("/api/acceptance/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 9 and body["all_pass"] is False
    assert len(body["blocks"]) == 9
    for b in body["blocks"]:
        assert set(b) == {"key", "no", "title", "page", "status", "headline", "note", "jobs"}
        assert b["status"] in ("pass", "fail", "missing")
        if b["status"] == "missing":
            assert not any(w in b["note"] for w in FAIL_WORDS), b["key"]
        assert set(b["jobs"]) <= set(jobs_core.JOBS), b["key"]


async def test_overview_missing_gates_are_missing_not_failed(client):
    body = (await client.get("/api/acceptance/overview")).json()
    by_key = {b["key"]: b for b in body["blocks"]}
    for key in ("golden", "corpus", "dataset", "train", "eval", "export", "scan", "pool"):
        assert by_key[key]["status"] == "missing", key
    # :8110 测试环境不在线 → 同样 missing(未拉起 ≠ 服务坏)
    assert by_key["classifier"]["status"] == "missing"


async def test_errors_missing_report_returns_empty(client):
    r = await client.get("/api/acceptance/errors")
    assert r.status_code == 200
    body = r.json()
    assert body["eval"]["present"] is False
    assert body["errors"] == []


async def test_data_missing_products_still_200(client):
    r = await client.get("/api/acceptance/data")
    assert r.status_code == 200
    body = r.json()
    assert len(body["topic_names"]) == 17
    assert body["dataset"]["clean"] is None          # 没有考卷时不许装干净
    assert body["model"]["present"] is False
    assert body["topics"]["present"] is False or body["topics"]["total"] == 0


async def test_eval_missing_reports(client):
    r = await client.get("/api/acceptance/eval")
    assert r.status_code == 200
    body = r.json()
    assert body["eval"]["present"] is False and body["scan"]["present"] is False
    assert body["threshold_in_use"] is None


async def test_overview_with_model_files_sizes_human_readable(client, tmp_path):
    """回归:训练三件套齐时 train 闸要算总大小——产物一直缺失时这分支从没跑过,曾因笔误 500。"""
    m = tmp_path / "model"
    m.mkdir()
    for name in ("model.safetensors", "tokenizer.json", "threshold.json"):
        (m / name).write_text("x" * 100, encoding="utf-8")
    r = await client.get("/api/acceptance/overview")
    assert r.status_code == 200
    train = next(b for b in r.json()["blocks"] if b["key"] == "train")
    assert train["status"] == "pass"
    assert "300 B" in train["headline"]


async def test_corrupt_report_reads_as_missing_not_500(client, tmp_path):
    """作业被页面 stop 掐在写一半时,产物 json 可能是残的——按没跑过处理,不炸页面。"""
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "eval_report.json").write_text('{"ran_at": "2026-09-', encoding="utf-8")
    r = await client.get("/api/acceptance/overview")
    assert r.status_code == 200
    by_key = {b["key"]: b for b in r.json()["blocks"]}
    assert by_key["eval"]["status"] == "missing"
    r2 = await client.get("/api/acceptance/errors")
    assert r2.status_code == 200 and r2.json()["eval"]["present"] is False
