# tests/test_tool_ticket.py
from app.db.models import Conversation, Ticket
from app.tools.business import create_ticket

def test_conversation_id_hidden_from_model_schema():
    # InjectedToolArg 的参数不出现在暴露给 LLM 的 tool-call schema(bind_tools 实际发的就是它);
    # get_input_schema() 是执行用完整 schema,injected 参数在设计上存在,不能作为断言对象
    props = create_ticket.tool_call_schema.model_json_schema().get("properties", {})
    assert "description" in props and "ticket_type" in props
    assert "conversation_id" not in props        # 关键:模型看不到会话主键

async def test_create_ticket_injects_conversation_id_and_writes(db_session_factory, db_clean):
    async with db_session_factory() as s:
        conv = Conversation(user_id="u1")
        s.add(conv)
        await s.commit()
        cid = conv.id
    r = await create_ticket.ainvoke(
        {"description": "商品损坏要退货", "ticket_type": "售后", "conversation_id": cid}
    )
    assert r["ticket_no"].startswith("T")
    async with db_session_factory() as s:
        t = await s.get(Ticket, r["ticket_no"])
        assert t is not None and t.conversation_id == cid
