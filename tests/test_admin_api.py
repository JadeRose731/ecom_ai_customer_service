# ch03 Task 17:后台聚合首页——依赖全挂也得出全部卡片,各自报自己读不到
import json

from app.db import repository


async def test_overview_returns_all_cards_even_when_deps_down(client, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("依赖挂了")

    for fn in ("knowledge_stats", "staging_stats", "list_conversations_with_messages"):
        monkeypatch.setattr(f"app.api.admin.{fn}", boom)
    monkeypatch.setattr("app.api.observability.list_eval_runs", boom)   # ch09:趋势块同款
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = r.json()["cards"]
    assert len(cards) == 6  # ch04 起 + RAG 评估卡 + ch09 观测卡
    by_key = {c["key"]: c for c in cards}
    # 依赖库的三张卡连坐报读不到;RAG 评估/观测卡只读文件,自己报自己的态
    for key in ("chat", "kb", "staging"):
        assert by_key[key]["state"] == "unreachable"
    assert by_key["rageval"]["state"] in ("empty", "ok")
    assert by_key["observability"]["state"] == "unreachable"   # list_eval_runs 挂 → 观测卡连坐


async def test_overview_reports_kb_numbers(client, db_session_factory, db_clean):
    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    cards = {c["key"]: c for c in r.json()["cards"]}
    assert cards["kb"]["state"] == "empty"  # 没数据是一种状态,不是错误


async def test_observability_card_ok_without_cost_data(client, tmp_path, monkeypatch):
    # ch09 review #4:成本账没跑过(top_share=None)不该把整卡炸成 unreachable
    from app.api import observability as obs_api
    from app.config import settings

    monkeypatch.setattr(obs_api, "_COST_JSON", tmp_path / "absent.json")   # 成本报告缺失
    cur = round(float(settings.evidence_confidence_threshold), 2)
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(
        {"recommended_threshold": cur, "youden_j": 0.5, "distribution": {}, "scan": []},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(obs_api, "_CAL_JSON", cal)

    r = await client.get("/api/admin/overview")
    assert r.status_code == 200
    card = {c["key"]: c for c in r.json()["cards"]}["observability"]
    assert card["state"] == "ok"                     # 校准同步即可 ok,不因成本缺数读数失败
    assert "最烧钱" not in card["conclusion"]        # 没有成本数据就不显示那半句
