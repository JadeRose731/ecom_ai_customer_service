# ch04:agent/chat 接线——citations 事件下发 + 证据不足拒答落池。
# query_faq 管线子步 mock,agent 编排走真的(FakeModel 模式同 test_agent_orchestration)。
import json

import pytest
from langchain_core.messages import AIMessage

from app.core import agent
from app.db import repository as repo


class _FakeToolRun:
    def __init__(self, name, payload, cid="tc1"):
        self.name = name
        self.tool_call_id = cid
        class _M:
            content = json.dumps(payload, ensure_ascii=False)
        self.tool_message = _M()


def test_extract_faq_citations():
    runs = [_FakeToolRun("query_faq", {"sufficient": True, "evidence": "[1] x",
            "citations": [{"n": 1, "id": 5, "section_path": "运费政策", "question": "运费", "answer": "满99包邮", "content_type": "faq"}]})]
    faq = agent._faq_result(runs)
    assert faq["sufficient"] is True and faq["citations"][0]["id"] == 5


def test_extract_faq_none_when_absent():
    runs = [_FakeToolRun("query_order", {"x": 1})]
    assert agent._faq_result(runs) is None


def test_extract_faq_none_on_bad_json():
    class _Bad:
        name = "query_faq"
        tool_call_id = "tc9"
        class _M:
            content = "not-json"
        tool_message = _M()
    assert agent._faq_result([_Bad()]) is None


class _ScriptedModel:
    def __init__(self, scripted, stream_tokens=None):
        self._scripted = list(scripted)
        self._stream_tokens = list(stream_tokens or [])
        self.bind_calls = 0
        self.astream_messages = []

    def bind_tools(self, tools):
        self.bind_calls += 1
        return self

    async def ainvoke(self, messages):
        return self._scripted.pop(0)

    async def astream(self, messages):
        self.astream_messages.append(list(messages))
        for t in self._stream_tokens:
            from langchain_core.messages import AIMessageChunk
            yield AIMessageChunk(content=t)


def _mock_pipeline(monkeypatch, sufficient=True):
    async def fake_understand(q): return {"standard": q, "expanded": []}
    monkeypatch.setattr("app.core.query_understanding.understand", fake_understand)
    if sufficient:
        async def fake_search(query, **kw):
            return [{"id": 7, "rerank_score": 0.9, "question": "运费怎么算", "answer": "满99包邮",
                     "section_path": "运费政策", "content_type": "faq", "category": "运费"}]
    else:
        async def fake_search(query, **kw): return []
    monkeypatch.setattr("app.core.retrieval.search_knowledge", fake_search)
    async def fake_check(q, texts): return {"useful": sufficient, "reason": "判" if not sufficient else "够"}
    monkeypatch.setattr("app.core.selfcheck.check_sufficient", fake_check)


@pytest.mark.asyncio
async def test_stream_emits_citations_and_swaps_rag_system(db_session_factory, db_clean, monkeypatch):
    _mock_pipeline(monkeypatch, sufficient=True)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "邮费多少"}, "id": "c1"}])
    model = _ScriptedModel([first], stream_tokens=["满99元包邮[1]。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "邮费多少", None, model=model)]
    types = [e["type"] for e in events]
    assert "citations" in types
    cit = next(e for e in events if e["type"] == "citations")
    assert cit["items"][0]["n"] == 1 and cit["items"][0]["section_path"] == "运费政策"
    # 收敛步的 system 换成带引用规则的 RAG system(证据足才换)
    conv_msgs = model.astream_messages[0]
    assert "引用" in conv_msgs[0].content
    # citations 事件先于 delta(前端先把引用数据挂上再渲染 [n] 角标)
    assert types.index("citations") < types.index("delta")


@pytest.mark.asyncio
async def test_stream_insufficient_pools_question_and_keeps_agent_system(
        db_session_factory, db_clean, monkeypatch):
    _mock_pipeline(monkeypatch, sufficient=False)
    calls = {}
    async def fake_pool(cid, raw, source, reason):
        calls.update(cid=cid, raw=raw, source=source, reason=reason)
        return 1
    monkeypatch.setattr("app.db.repository.insert_low_confidence", fake_pool)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "火星车怎么买"}, "id": "c1"}])
    model = _ScriptedModel([first], stream_tokens=["暂时没有查到相关信息。"])
    events = [ev async for ev in agent.stream_agent_turn("u1", "你们卖火星车吗", None, model=model)]
    assert all(e["type"] != "citations" for e in events)
    assert calls["raw"] == "你们卖火星车吗" and calls["source"] == "retrieval_low_conf"
    conv_msgs = model.astream_messages[0]
    assert "引用" not in conv_msgs[0].content  # 证据不足不换 RAG system,靠拒答信号


@pytest.mark.asyncio
async def test_non_stream_result_carries_citations(db_session_factory, db_clean, monkeypatch):
    _mock_pipeline(monkeypatch, sufficient=True)
    first = AIMessage(content="", tool_calls=[
        {"name": "query_faq", "args": {"keyword": "邮费多少"}, "id": "c1"}])
    model = _ScriptedModel([first, AIMessage(content="满99元包邮[1]。")])
    res = await agent.run_agent_turn("u1", "邮费多少", None, model=model)
    assert res.citations and res.citations[0]["id"] == 7
