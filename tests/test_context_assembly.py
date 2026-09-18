from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.prompts import AGENT_SYSTEM
from app.graph.nodes import _agent_messages, _history_text

def _state(n_turns=3, summary="", upto=0):
    msgs = []
    db_id = 1
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return {"messages": msgs, "summary": summary, "summary_upto_msg_id": upto}

def test_agent_messages_order_with_summary():
    """拼装:人设 system 打头 → 滑窗原文;摘要随本轮材料跟在最后一条用户消息后。
    摘要不进 system:上游 chat template 会把列表里所有 system 上提合并,工具 schema
    被挤到可变内容之后,前缀缓存全 miss(实测 cache_read 2048→0),故走用户侧材料消息。"""
    ms = _agent_messages(_state(summary="用户问过订单1001", upto=0))
    assert isinstance(ms[0], SystemMessage) and "小喵" in ms[0].content   # 人设红线打头
    assert isinstance(ms[1], HumanMessage)                     # 滑窗原文紧随其后(非第二条 system)
    assert not any(isinstance(m, SystemMessage) for m in ms[1:])  # 全列表只有一条 system
    assert isinstance(ms[-1], AIMessage) and ms[-1].content == "回答2"
    assert "订单1001" in ms[-2].content                        # 摘要材料插在最后一条用户消息后

def test_agent_messages_no_summary_no_extra_system():
    ms = _agent_messages(_state(summary="", upto=0))
    assert isinstance(ms[0], SystemMessage)
    assert not isinstance(ms[1], SystemMessage)     # 无摘要不插第二条 system

def test_agent_messages_window_cut_by_boundary():
    ms = _agent_messages(_state(n_turns=6, summary="早期摘要", upto=6))
    win = [m for m in ms if not isinstance(m, SystemMessage)]
    assert win[0].id == "db-7"                      # 滑窗从边界后第一条用户消息接原文

# 前缀缓存不变量:system 恒为 AGENT_SYSTEM,可变内容一律不许拼进去。
# 注释会被无视,这条测试不会——后面几章谁想往 system 里塞东西,这里立刻红。
def test_system_message_is_always_the_static_prompt():
    for state in ({"route": "business", "messages": [HumanMessage("订单1001到哪了")]},
                  {"route": "knowledge", "evidence": "[1] 退货: 7天",
                   "messages": [HumanMessage("能退吗")]}):
        assert _agent_messages(state)[0].content == AGENT_SYSTEM

# 插入位置:第 1 步的整段 prompt 必须是第 2 步的严格前缀,ReAct 轮内才命中
def test_step1_prompt_is_strict_prefix_of_step2():
    base = {"route": "knowledge", "evidence": "[1] 退货: 7天"}
    q = HumanMessage("能退吗", id="db-1")
    s1 = _agent_messages({**base, "messages": [q]})
    s2 = _agent_messages({**base, "messages": [q, AIMessage("在查")]})
    shape = lambda ms: [(m.type, m.content) for m in ms]      # noqa: E731
    assert shape(s2)[:len(s1)] == shape(s1)

def test_history_text_prepends_summary_and_windows():
    text = _history_text(_state(n_turns=6, summary="用户问过订单1001", upto=6))
    assert text.startswith("(早前对话摘要:用户问过订单1001")
    assert "问题0" not in text                       # 边界前原文不出现
    assert "问题3" in text                           # 窗内原文在(排除当前最后一条 human)

def test_history_text_no_summary_same_as_before():
    text = _history_text(_state(n_turns=2))
    assert "摘要" not in text and "问题0" in text
