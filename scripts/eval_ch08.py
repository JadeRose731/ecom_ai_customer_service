"""ch08 验收样例端到端评估(建工单确认流 + MCP 工具实调)。需全服务起(make dev,含 mcp-up)。
用法:PYTHONPATH=. uv run python scripts/eval_ch08.py
样例:①缺描述追问不弹卡 ②补齐描述弹预览卡 ③确认→工单落库 ④取消→权限拒绝审计 ⑤MCP 物流实调。
退出码非 0 表示有 FAIL。"""
import asyncio
import re

import httpx
from sqlalchemy import func, select

from app.db.base import async_session
from app.db.models import Ticket, ToolAuditLog

BASE = "http://localhost:8000"
TICKET_NO = re.compile(r"T\d{6,}")


async def tickets_count() -> int:
    async with async_session() as s:
        return (await s.execute(select(func.count()).select_from(Ticket))).scalar_one()


async def last_audit(conv: int, tool: str, status: str | None = None):
    q = select(ToolAuditLog).where(ToolAuditLog.conversation_id == conv,
                                   ToolAuditLog.tool_name == tool).order_by(ToolAuditLog.id.desc())
    if status:
        q = q.where(ToolAuditLog.status == status)
    async with async_session() as s:
        return (await s.execute(q.limit(1))).scalar_one_or_none()


async def agent(client, msg, cid=None) -> dict:
    r = await client.post(f"{BASE}/api/agent",
                          json={"user_id": "eval-ch08", "message": msg, "conversation_id": cid})
    return r.json()


async def resume(client, cid: int, extra: dict) -> str:
    """POST /api/actions/resume(SSE);返回整个 body 文本供断言(delta/工单号在里面)。"""
    r = await client.post(f"{BASE}/api/actions/resume", json={"conversation_id": cid, **extra})
    return r.text


async def main():
    async with httpx.AsyncClient(timeout=300) as c:
        results = []

        # ① 缺描述:模型被追问,不弹 confirm_ticket 卡、tickets 不落
        n0 = await tickets_count()
        b = await agent(c, "帮我建个工单")
        cid1 = b["conversation_id"]
        n0b = await tickets_count()
        results.append(("①缺描述→不弹卡且不落库", b.get("interrupt") is None and n0b == n0,
                        {"interrupt": b.get("interrupt"), "answer": (b.get("answer") or "")[:40]}))

        # ② 补齐描述:弹预览卡且两字段齐
        b = await agent(c, "问题描述是猫砂盆漏电", cid1)
        intr = b.get("interrupt") or {}
        pv = intr.get("preview") or {}
        results.append(("②补齐描述→confirm_ticket 预览字段齐",
                        intr.get("type") == "confirm_ticket" and bool(pv.get("description"))
                        and bool(pv.get("ticket_type")),
                        {"preview": pv}))

        # ③ 确认提交:tickets +1,续流答复含工单号
        body = await resume(c, cid1, {"confirmed": True})
        n1 = await tickets_count()
        results.append(("③确认→工单落库+答复含工单号",
                        n1 == n0 + 1 and TICKET_NO.search(body) is not None,
                        {"tickets": (n0, n1), "body≈": body[:60]}))

        # ④ 取消:无新工单,审计落权限拒绝
        n2 = await tickets_count()
        b = await agent(c, "再帮我建个工单,猫抓板被抓烂了")
        cid2 = b["conversation_id"]
        intr = b.get("interrupt") or {}
        ok4 = intr.get("type") == "confirm_ticket"
        detail4 = {"interrupt_type": intr.get("type")}
        if ok4:
            await resume(c, cid2, {"confirmed": False})
            a = await last_audit(cid2, "create_ticket", "权限拒绝")
            ok4 = (await tickets_count()) == n2 and a is not None
            detail4["audit"] = "权限拒绝" if a is not None else "缺"
        results.append(("④取消→无新工单+审计权限拒绝", ok4, detail4))

        # ⑤ MCP 物流实调:query_logistics 经引擎成功,答复非空
        b = await agent(c, "我的猫粮什么时候到,单号 SF123456789")
        cid3 = b["conversation_id"]
        a = await last_audit(cid3, "query_logistics", "成功")
        ok5 = a is not None and a.tool_source == "mcp" and bool(b.get("answer"))
        results.append(("⑤MCP query_logistics 实调(mcp/成功)", ok5,
                        {"audit": None if a is None else f"{a.tool_source}/{a.status}/{a.duration_ms}ms",
                         "answer": (b.get("answer") or "")[:40]}))

        for name, ok, detail in results:
            print(f"{'✅' if ok else '❌'} {name} -> {detail}")

    failed = sum(1 for _, ok, _ in results if not ok)
    print("\n提示:⑤失败多半是 MCP Server 没起(`make mcp-up`)——Server 全挂时其工具本轮不可见;")
    print("①②③④失败先看 app 日志 `ch05 turn ... trace=...` 与工具审计行。")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
