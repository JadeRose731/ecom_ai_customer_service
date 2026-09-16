from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.routing import confidence_gate, route_by_intent, should_continue
from app.graph.state import ConversationState


def _builder() -> StateGraph:
    b = StateGraph(ConversationState)
    # 节点
    b.add_node("resolve_reference", nodes.resolve_reference)
    b.add_node("classify_intent", nodes.classify_intent)
    b.add_node("forced_rag", nodes.forced_rag)
    b.add_node("confidence_check", nodes.confidence_check)  # 实体节点(trace 门控);判断在其后条件边
    b.add_node("agent_llm", nodes.agent_llm)
    b.add_node("agent_tools", nodes.agent_tools)
    b.add_node("complaint_reply", nodes.complaint_reply)
    b.add_node("chitchat_reply", nodes.chitchat_reply)
    b.add_node("fallback_reply", nodes.fallback_reply)
    b.add_node("log", nodes.log_node)

    # 骨架:消解 → 意图 → 分流
    b.add_edge(START, "resolve_reference")
    b.add_edge("resolve_reference", "classify_intent")
    b.add_conditional_edges("classify_intent", route_by_intent, {
        "knowledge": "forced_rag",
        "business": "agent_llm",
        "complaint": "complaint_reply",
        "chitchat": "chitchat_reply",
    })
    # 知识路:检索 → 生成前证据闸
    b.add_edge("forced_rag", "confidence_check")
    b.add_conditional_edges("confidence_check", confidence_gate, {
        "strong": "agent_llm",
        "weak": "fallback_reply",
    })
    # 主力 Agent ReAct 环
    b.add_conditional_edges("agent_llm", should_continue, {
        "continue": "agent_tools",
        "stop": "log",
    })
    b.add_edge("agent_tools", "agent_llm")
    # 确定性出口 → 日志 → END
    b.add_edge("complaint_reply", "log")
    b.add_edge("chitchat_reply", "log")
    b.add_edge("fallback_reply", "log")
    b.add_edge("log", END)
    return b


def build_graph(checkpointer=None):
    return _builder().compile(checkpointer=checkpointer)
