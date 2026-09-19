"""ch09 Cost Control:按意图统计 token 花销(README:「按意图把 token 分堆一算,
哪类意图最烧钱立马现形」)。数据源 = Langfuse Metrics API(意图在 Task 3 打成 intent:xxx tag,
产生模型调用的节点入口各自携带,LangGraph 节点 task 边界下逐节点打标)。
运行:make cost-report(DAYS=N 窗口天数,默认 7;需 Langfuse 在跑且 .env 配好三变量)。
产物:dev-notes/ch09-cost-report.txt
说明:自定义模型名(glm 系列)在 Langfuse 无内置单价,统计以 token 数为准;
要看钱在 Langfuse 界面配模型单价即可,不在本脚本范围。

v4 实测口径(偏差③/⑦,dev-notes 有全记录,2026-09-19 对活服务 v4.38.0 逐条验证):
SDK 方法是 api.metrics.metrics(query=json.dumps({...}))(底层 GET api/public/v2/metrics,
query 作 URL 参数);events_only 模式只有 observations 视图(traces 视图 404);
返回 .data 是 list[dict],行键 = 维度字段 + {measure}_{aggregation}(sum_totalTokens/count_count);
tags 过滤必须 type: "arrayOptions" + operator: "any of" + 数组值。
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone

from app.core.intent import INTENTS
from app.core.observability import get_langfuse

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_OUT = _ROOT / "dev-notes/ch09-cost-report.txt"


def _window(days: int) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return (now - timedelta(days=days)).strftime(fmt), now.strftime(fmt)


def _intent_of(tags) -> str | None:
    """行上的 tags 是列表;取第一个 intent: 前缀 tag(本系统每个 span 只打一个)。"""
    for t in tags or []:
        if isinstance(t, str) and t.startswith("intent:"):
            return t.removeprefix("intent:")
    return None


def _query_grouped_by_tags(client, frm: str, to: str) -> list[dict] | None:
    """主路:observations 视图(仅 GENERATION)按 tags 维度分组,一趟拿全部分账。
    返回 None 表示不可用,走逐意图回退。"""
    try:
        resp = client.api.metrics.metrics(query=json.dumps({
            "view": "observations",
            "dimensions": [{"field": "tags"}],
            "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                        {"measure": "count", "aggregation": "count"}],
            "filters": [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"}],
            "fromTimestamp": frm, "toTimestamp": to,
        }))
        rows = []
        for row in resp.data:
            intent = _intent_of(row.get("tags"))
            if intent is None:
                continue   # 无 tag 的 generation(飞轮/评估脚本等图外调用)不进意图账
            rows.append({"intent": intent,
                         "tokens": int(float(row.get("sum_totalTokens") or 0)),
                         "count": int(float(row.get("count_count") or 0))})
        return rows
    except Exception as e:  # noqa: BLE001 —— 主路挂了回退兜,脚本不崩
        print(f"[主路 tags 分组不可用:{type(e).__name__},走逐意图回退] {e}")
        return None


def _query_per_intent(client, frm: str, to: str) -> list[dict]:
    """回退:逐意图 tag 过滤聚合(每类一次查询,不分组)。"""
    rows = []
    for intent in INTENTS:
        resp = client.api.metrics.metrics(query=json.dumps({
            "view": "observations",
            "metrics": [{"measure": "totalTokens", "aggregation": "sum"},
                        {"measure": "count", "aggregation": "count"}],
            "filters": [{"column": "type", "operator": "=", "value": "GENERATION", "type": "string"},
                        {"column": "tags", "operator": "any of",
                         "value": [f"intent:{intent}"], "type": "arrayOptions"}],
            "fromTimestamp": frm, "toTimestamp": to,
        }))
        tokens = count = 0
        for row in resp.data:
            tokens += int(float(row.get("sum_totalTokens") or 0))
            count += int(float(row.get("count_count") or 0))
        if count:
            rows.append({"intent": intent, "tokens": tokens, "count": count})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    client = get_langfuse()
    if client is None:
        print("Langfuse 未配置(.env 三变量),无账可查。")
        return 1
    frm, to = _window(args.days)
    rows = _query_grouped_by_tags(client, frm, to)
    if rows is None:
        rows = _query_per_intent(client, frm, to)
    # 同一意图可能多行(tags 维度的行键是完整 tag 集)——按意图归并再排
    merged: dict[str, dict] = {}
    for r in rows:
        m = merged.setdefault(r["intent"], {"intent": r["intent"], "tokens": 0, "count": 0})
        m["tokens"] += r["tokens"]
        m["count"] += r["count"]
    rows = sorted(merged.values(), key=lambda r: r["tokens"], reverse=True)

    total = sum(r["tokens"] for r in rows) or 1
    lines = [f"=== 按意图 token 花销(近 {args.days} 天,数据源 Langfuse)===",
             f"{'意图':6s} {'请求数':>8s} {'总tokens':>12s} {'平均tokens':>12s} {'占比':>7s}"]
    for i, r in enumerate(rows):
        mark = "  ← 最烧钱" if i == 0 else ""
        lines.append(f"{r['intent']:6s} {r['count']:>8d} {r['tokens']:>12,d} "
                     f"{r['tokens'] // max(r['count'], 1):>12,d} {r['tokens'] / total:>6.0%}{mark}")
    if not rows:
        lines.append("(窗口内没有带 intent tag 的 generation——先聊几句再来)")
    out = "\n".join(lines)
    print(out)
    _OUT.write_text(out + "\n", encoding="utf-8")
    print(f"\n报告已落 {_OUT.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
