from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.memory import SessionStore, build_window, summary_line, summary_system, trim_history

def test_store_get_unknown_session_returns_empty():
    assert SessionStore().get("nope") == []

def test_store_append_and_get_isolated_by_session():
    store = SessionStore()
    store.append("a", HumanMessage("hi"), AIMessage("hello"))
    store.append("b", HumanMessage("嗨"))
    assert len(store.get("a")) == 2
    assert len(store.get("b")) == 1

def test_trim_keeps_recent_and_starts_on_human():
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(f"问题{i}:" + "喵" * 50))
        msgs.append(AIMessage(f"回答{i}:" + "喵" * 50))
    trimmed = trim_history(msgs, max_tokens=200)
    assert 0 < len(trimmed) < len(msgs)
    assert trimmed[0].type == "human"
    assert trimmed[-1] == msgs[-1]

def test_trim_noop_when_under_budget():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    assert trim_history(msgs, max_tokens=2000) == msgs


# ---- ch07 锚点切窗 + 摘要注入 helpers ----

def _dialog(n_turns: int, start_db_id: int = 1) -> list:
    """n 轮对话,用户消息带 db-id 锚点(每轮占 2 个 id:user、assistant)。"""
    msgs = []
    db_id = start_db_id
    for i in range(n_turns):
        msgs.append(HumanMessage(f"问题{i}", id=f"db-{db_id}"))
        msgs.append(AIMessage(f"回答{i}"))
        db_id += 2
    return msgs

def test_build_window_cuts_at_anchor():
    msgs = _dialog(10)                       # 用户消息 db-1,3,...,19
    win = build_window(msgs, summary_upto_msg_id=8, max_tokens=100000)
    assert win[0].id == "db-9"               # 第一条 > 8 的用户消息
    assert len(win) == 12                    # 第 5-10 轮共 6 轮
    assert win[-1] == msgs[-1]

def test_build_window_no_boundary_keeps_all():
    msgs = _dialog(3)
    assert build_window(msgs, summary_upto_msg_id=0, max_tokens=100000) == msgs

def test_build_window_anchor_missing_degrades_to_trim():
    msgs = [HumanMessage("旧消息无锚点"), AIMessage("答")] * 3   # ch07 前的旧会话消息
    win = build_window(list(msgs), summary_upto_msg_id=99, max_tokens=100000)
    assert len(win) == 6                     # 找不到锚点 → 不切,整段进 trim

def test_build_window_mixed_history_keeps_unanchored_tail():
    """ch07 前的旧消息无锚点、紧贴首个锚点之前的尾巴必须留在窗里:
    这段在摘要里也没盖住(摘要段只收边界之前的 id),锚点一硬切就两头落空。"""
    legacy = [HumanMessage("旧问题0"), AIMessage("旧回答0"),
              HumanMessage("旧问题1"), AIMessage("旧回答1")]
    msgs = legacy + _dialog(3, start_db_id=9)    # 首个锚点 db-9 在 index 4
    win = build_window(msgs, summary_upto_msg_id=8, max_tokens=100000)
    assert win[0] == legacy[0]                   # 未锚定尾巴回卷进窗
    assert win[-1] == msgs[-1]

def test_build_window_mixed_history_stops_at_older_anchor():
    """回卷不越过更早的锚点:边界前的锚定轮(db-1)已进摘要,不回窗;
    其未锚定 AI 回复(id≤边界)也已被摘要覆盖,trim 落在首个 human 上裁掉属正确语义。"""
    msgs = _dialog(1, start_db_id=1) + _dialog(1, start_db_id=3) + _dialog(1, start_db_id=5)
    # [user db-1, ai, user db-3, ai, user db-5, ai]
    win = build_window(msgs, summary_upto_msg_id=2, max_tokens=100000)
    assert win[0].id == "db-3"                   # 窗从首个 > 边界的锚点起(trim 对齐 human)
    assert not any(m.id == "db-1" for m in win)  # 不越过更早的锚点 db-1

def test_build_window_token_cap_still_applies():
    msgs = _dialog(20)
    for m in msgs:
        m.content = m.content + "喵" * 200   # 撑大 token
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=300)
    assert 0 < len(win) < len(msgs)
    assert win[0].type == "human"            # trim start_on=human

def test_build_window_never_returns_empty():
    msgs = [HumanMessage("超长" + "喵" * 5000, id="db-1")]
    win = build_window(msgs, summary_upto_msg_id=0, max_tokens=10)
    assert win == msgs                       # 裁到空则宁可超预算也回退原窗

def test_summary_helpers():
    assert summary_line(None) == "" and summary_line("") == ""
    assert "订单1001" in summary_line("用户问过订单1001")
    assert summary_system(None) is None
    ss = summary_system("用户问过订单1001")
    assert isinstance(ss, SystemMessage) and "订单1001" in ss.content
