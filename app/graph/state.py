from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_dict(a: dict | None, b: dict | None) -> dict:
    """trace 累加 reducer:后写覆盖同键,其余合并。"""
    return {**(a or {}), **(b or {})}


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # 跨轮历史,checkpointer 续接
    user_id: str
    conversation_id: int
    intent: str            # 八类之一(含「其他」)
    resolved_query: str    # 指代消解+改写后的完整问句(下游检索/判意图都用它)
    intent_confidence: float  # 意图 JSON 的 confidence(0-1)
    order_id: str          # refund_flow:抽到/点选回填的订单号
    order_data: dict       # refund_flow:query_order 查到的订单数据
    route: str             # escalate | fallback_script | knowledge | refund_flow | business
    evidence: str          # 知识路编号证据文本
    citations: list        # 引用 chunk(前端可点)
    evidence_strong: bool  # 生成前证据闸信号
    answer: str            # 确定性节点产出的答复(agent 答复走流式,不落此字段)
    steps: int             # ReAct 步数(停止条件)
    tokens_used: int       # 逐步累加进 trace,供排障与 ch09 统计,不当停止条件
    suggested_actions: list  # [{"type": "transfer_human"} | {"type": "create_ticket", "draft": {...}}]
    trace: Annotated[dict, merge_dict]  # 留痕
