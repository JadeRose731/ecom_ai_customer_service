"""ch09 观测挂载:缺配置不挂回调、不碰 langfuse;齐配置才 with_config;
tag_intent 双通道直改 root span + OTel attach(README 的 update_current_trace 已删、
propagate_attributes CM 丢弃即被 GC 撤销 attach,考证记 dev-notes)。"""
from uuid import uuid4

import pytest
from opentelemetry import context as otel_context_api

from app.core import observability


@pytest.fixture(autouse=True)
def _restore_otel_context():
    """tag_intent 会 attach 进当前 OTel 上下文,测完恢复,不跨用例泄漏。"""
    snapshot = otel_context_api.get_current()
    yield
    otel_context_api.attach(snapshot)


def _enabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "pk")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://localhost:3000")


def test_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    monkeypatch.setattr("app.config.settings.langfuse_secret_key", "sk")
    monkeypatch.setattr("app.config.settings.langfuse_base_url", "http://x")
    assert observability.langfuse_enabled() is False


def test_attach_returns_graph_unchanged_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    sentinel = object()
    assert observability.attach_observability(sentinel) is sentinel


def test_tag_intent_noop_when_disabled(monkeypatch):
    monkeypatch.setattr("app.config.settings.langfuse_public_key", "")
    observability.tag_intent("商品咨询", 0.9)   # 不许抛异常、不许要求 langfuse 已装好服务


def test_attach_wraps_with_callbacks_when_enabled(monkeypatch):
    _enabled(monkeypatch)

    calls = {}

    class FakeGraph:
        def with_config(self, cfg):
            calls["cfg"] = cfg
            return "wrapped"

    monkeypatch.setattr(observability, "_init_client", lambda: object())
    monkeypatch.setattr(observability, "_make_handler", lambda: "HANDLER")
    assert observability.attach_observability(FakeGraph()) == "wrapped"
    assert calls["cfg"] == {"callbacks": ["HANDLER"]}


class _FakeSpan:
    def __init__(self, session_id=None):
        self.attributes = {} if session_id is None else {"session.id": session_id}
        self.set_calls = []

    def set_attribute(self, key, value):
        self.set_calls.append((key, value))


def _fake_handler(monkeypatch, span):
    root_id = uuid4()

    class FakeObs:
        _otel_span = span

    class FakeHandler:
        _root_run_states = {root_id: object()}
        _runs = {root_id: FakeObs()}

    monkeypatch.setattr(observability, "_handler", FakeHandler())
    return span


def test_tag_intent_sets_trace_tags_and_propagates(monkeypatch):
    """enabled 时双通道:直改 root span 的 langfuse.trace.tags(观测页 trace 显示)+
    attach 进 OTel 上下文(本 task 内后续 observation 逐个带 tag)。"""
    _enabled(monkeypatch)
    span = _fake_handler(monkeypatch, _FakeSpan(session_id="c1"))

    observability.tag_intent("商品咨询", 0.87, session_id="c1")
    assert span.set_calls == [("langfuse.trace.tags", ["intent:商品咨询"])]
    assert otel_context_api.get_value("langfuse.propagated.tags") == ["intent:商品咨询"]


def test_tag_intent_skips_other_session(monkeypatch):
    """并发轮次不串:session_id 不匹配的活跃 trace 不动。"""
    _enabled(monkeypatch)
    span = _fake_handler(monkeypatch, _FakeSpan(session_id="别的会话"))
    observability.tag_intent("商品咨询", 0.87, session_id="c1")
    assert span.set_calls == []


def test_tag_intent_noop_when_handler_absent(monkeypatch):
    """enabled 但图未挂 handler(如单测环境只初始化了 client)时安全跳过。"""
    _enabled(monkeypatch)
    monkeypatch.setattr(observability, "_handler", None)
    observability.tag_intent("商品咨询", 0.87)   # 不许抛
    assert otel_context_api.get_value("langfuse.propagated.tags") == ["intent:商品咨询"]


def test_tag_intent_swallows_failure(monkeypatch):
    """任一通道抛异常(root span 属性写入/attach)只吞掉,不许影响业务节点。"""
    _enabled(monkeypatch)

    class BoomSpan:
        attributes = {"session.id": "c1"}

        def set_attribute(self, key, value):
            raise RuntimeError("otel 属性写入异常")

    _fake_handler(monkeypatch, BoomSpan())

    import opentelemetry.context

    def boom(*a, **kw):
        raise RuntimeError("attach 异常")
    monkeypatch.setattr(opentelemetry.context, "attach", boom)

    observability.tag_intent("商品咨询", 0.87, session_id="c1")   # 不许抛
