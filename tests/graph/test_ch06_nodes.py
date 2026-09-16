import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import nodes


@pytest.mark.asyncio
async def test_classify_intent_writes_confidence_and_route(monkeypatch):
    async def fake_classify(query, history=""):
        return {"intent": "退款退货", "confidence": 0.83}
    monkeypatch.setattr(nodes.intent_mod, "classify", fake_classify)
    out = await nodes.classify_intent({"messages": [HumanMessage("这个能退吗")],
                                       "resolved_query": "蓝牙耳机还能申请退货吗"})
    assert out["intent"] == "退款退货"
    assert out["intent_confidence"] == 0.83
    assert out["route"] == "refund_flow"                 # 退款退货 → refund_flow
    assert out["trace"]["route"] == "refund_flow"
    assert out["trace"]["intent_confidence"] == 0.83     # 不覆盖 confidence_check 的 confidence 键


def test_get_chat_model_honors_model_override():
    from app.core.llm import get_chat_model
    m = get_chat_model(model="glm-4-flash")
    assert m.model_name == "glm-4-flash"
    d = get_chat_model()
    from app.config import settings
    assert d.model_name == settings.chat_model
