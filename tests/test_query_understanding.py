# Query 理解的 prompt 质量走标注样例验证(tests/data/query_rewrite_samples.jsonl,
# 需真实聊天上游,见 make eval-rewrite);这里只锁结构契约:扁平字段、异常兜底。
import pytest

from app.core import query_understanding as qu


class _FakeRewrite:
    standard = "退货运费谁承担"
    expanded = ["邮费", "运费"]


def _fake_model():
    class _M:
        def with_structured_output(self, _schema):
            return object()  # 链整体被替换,这个对象不会被真正调用
    return _M()


@pytest.mark.asyncio
async def test_understand_returns_flat_fields(monkeypatch):
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, _): return _FakeRewrite()
    monkeypatch.setattr("app.core.query_understanding.get_chat_model",
                        lambda streaming=False: _fake_model())
    monkeypatch.setattr("app.core.query_understanding.QUERY_REWRITE_PROMPT",
                        Chain())
    out = await qu.understand("退货运费谁承担")
    assert out == {"standard": "退货运费谁承担", "expanded": ["邮费", "运费"]}


@pytest.mark.asyncio
async def test_understand_falls_back_to_original_when_empty(monkeypatch):
    class _Empty:
        standard = ""
        expanded = None
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, _): return _Empty()
    monkeypatch.setattr("app.core.query_understanding.get_chat_model",
                        lambda streaming=False: _fake_model())
    monkeypatch.setattr("app.core.query_understanding.QUERY_REWRITE_PROMPT",
                        Chain())
    out = await qu.understand("坏了咋整")
    assert out == {"standard": "坏了咋整", "expanded": []}
