from langchain_core.messages import HumanMessage

from app.core.prompts import (
    AGENT_PROMPT, AGENT_SYSTEM, CUSTOMER_SERVICE_PROMPT, EXTRACT_PROMPT,
    EXPAND_QUERIES_PROMPT, INTENT_CLASSIFY_PROMPT,
)

def test_customer_service_prompt_renders_with_history():
    msgs = CUSTOMER_SERVICE_PROMPT.format_messages(
        history=[HumanMessage("你们卖猫粮吗?")]
    )
    assert msgs[0].type == "system"
    assert msgs[-1].content == "你们卖猫粮吗?"
    # 行为约束必须写进 system prompt
    for keyword in ("客服", "不", "订单"):
        assert keyword in msgs[0].content

def test_extract_prompt_renders_text():
    msgs = EXTRACT_PROMPT.format_messages(text="订单 MH1 坏了要退款")
    assert msgs[0].type == "system"
    assert "订单 MH1 坏了要退款" in msgs[-1].content

def test_agent_system_covers_tool_principles():
    for kw in ["小喵", "工具", "query_faq", "create_ticket", "不要臆造"]:
        assert kw in AGENT_SYSTEM

def test_agent_prompt_has_history_placeholder():
    assert any(getattr(m, "variable_name", None) == "history"
               for m in AGENT_PROMPT.messages)


def test_intent_classify_prompt_renders_json_example():
    # 防「裸 JSON 地雷」回退:示例 {} 一旦忘了转义成 {{}},模板会把 {"intent" 当变量,
    # 渲染即 KeyError(INVALID_PROMPT_INPUT),classify 被兜底无声掩盖。
    msgs = INTENT_CLASSIFY_PROMPT.format_messages(history=[], query="订单1001能退吗")
    assert INTENT_CLASSIFY_PROMPT.input_variables == ["history", "query"]
    assert msgs[0].type == "system"
    assert '{"intent"' in msgs[0].content


def test_expand_queries_prompt_renders_json_example():
    msgs = EXPAND_QUERIES_PROMPT.format_messages(query="蓝牙耳机怎么退货")
    assert EXPAND_QUERIES_PROMPT.input_variables == ["query"]
    assert msgs[0].type == "system"
    assert '{"queries"' in msgs[0].content
