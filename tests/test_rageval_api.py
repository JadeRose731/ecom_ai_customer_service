# ch04 Task 16:RAG 评估 API——只读产物,一个指标都不在 web 层重算。
# 口径:缺失=200+present=false(不是 500);在=与文件一字不差;best=总体 MRR 最高;
# generation=null 照样端检索段;坏 json 按没跑过处理。
import json

from app.api import rageval


def _report(tmp_path, monkeypatch, report: dict | None, raw: bytes | None = None):
    """把 REPORT_PATH 指到临时文件,写好(或写坏)产物后返回路径。"""
    p = tmp_path / "rag_eval.json"
    if raw is not None:
        p.write_bytes(raw)
    elif report is not None:
        p.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(rageval, "REPORT_PATH", p)
    return p


def _full_report(vector_mrr: float = 0.42) -> dict:
    return {
        "meta": {"ts": "2026-09-09 12:00:00", "k": 10, "strategies": ["vector", "hybrid_rerank"],
                 "n_samples": 80, "generation_done": True},
        "retrieval": {
            "vector": {"all": {"recall": 0.75, "mrr": vector_mrr},
                       "A_policy": {"recall": 0.8, "mrr": 0.7},
                       "B_model": {"recall": 0.6, "mrr": 0.4},
                       "C_colloquial": {"recall": 0.85, "mrr": 0.16}},
            "hybrid_rerank": {"all": {"recall": 0.93, "mrr": 0.88},
                              "A_policy": {"recall": 0.95, "mrr": 0.9},
                              "B_model": {"recall": 0.9, "mrr": 0.87},
                              "C_colloquial": {"recall": 0.94, "mrr": 0.86}},
        },
        "evidence_coverage": {"vector": 0.62, "hybrid_rerank": 0.81},
        "generation": {"done": True, "answer_coverage": {"hybrid_rerank": {"all": 0.77}},
                       "faithfulness": {"rate": 0.98, "n": 60},
                       "refusal": {"rate": 0.95, "refused": 19, "total": 20, "detail": []}},
    }


async def test_missing_report_is_200_present_false(client, tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, None)
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is False
    assert body["job"]["name"] == "eval-rag"          # 告诉页面该按哪个作业
    assert body["best"] is None


async def test_report_passes_through_unchanged(client, tmp_path, monkeypatch):
    report = _full_report()
    _report(tmp_path, monkeypatch, report)
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is True
    assert body["retrieval"] == report["retrieval"]                # 一字不差,不重算
    assert body["evidence_coverage"] == report["evidence_coverage"]
    assert body["generation"] == report["generation"]
    assert body["meta"] == report["meta"]
    assert body["generation_done"] is True


async def test_best_follows_overall_mrr(client, tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, _full_report())
    r = await client.get("/api/rag-eval/overview")
    assert r.json()["best"] == "hybrid_rerank"

    # 把 vector 的总体 MRR 改高,结论跟着换
    report = _full_report(vector_mrr=0.99)
    _report(tmp_path, monkeypatch, report)
    r = await client.get("/api/rag-eval/overview")
    assert r.json()["best"] == "vector"


async def test_generation_none_still_serves_retrieval(client, tmp_path, monkeypatch):
    report = _full_report()
    report["generation"] = None
    report["meta"]["generation_done"] = False
    _report(tmp_path, monkeypatch, report)
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval"] == report["retrieval"]                # 检索段照样端出来
    assert body["generation"] is None
    assert body["generation_done"] is False                        # 页面据此标「本次未完成」


async def test_corrupt_json_treated_as_never_run(client, tmp_path, monkeypatch):
    _report(tmp_path, monkeypatch, None, raw=b'{"meta": {"ts": "2026-09-09")')
    r = await client.get("/api/rag-eval/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["present"] is False
    assert body["best"] is None
