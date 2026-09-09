import pytest

from app.core import selfcheck


@pytest.mark.asyncio
async def test_check_sufficient_parses_flat(monkeypatch):
    class Fake:
        useful = False
        reason = "证据只讲运费,未涉及型号"
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, _): return Fake()
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("Pro型号功能", ["满99包邮"])
    assert out == {"useful": False, "reason": "证据只讲运费,未涉及型号"}


@pytest.mark.asyncio
async def test_check_sufficient_empty_evidence_still_judges(monkeypatch):
    seen = {}
    class Fake:
        useful = False
        reason = "无证据"
    class Chain:
        def __or__(self, o): return self
        async def ainvoke(self, kw):
            seen.update(kw)
            return Fake()
    monkeypatch.setattr("app.core.selfcheck._chain", lambda: Chain())
    out = await selfcheck.check_sufficient("q", [])
    assert out == {"useful": False, "reason": "无证据"}
    assert seen["evidence"] == "(无证据)"
