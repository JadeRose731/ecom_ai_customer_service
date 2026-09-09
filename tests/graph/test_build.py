from app.graph.build import _builder


def test_graph_has_all_nodes():
    g = _builder().compile()
    names = set(g.get_graph().nodes)
    for n in ["coref", "classify_intent", "forced_rag", "confidence_check",
              "agent_llm", "agent_tools", "complaint_reply", "chitchat_reply",
              "fallback_reply", "log"]:
        assert n in names, f"缺节点 {n}"


def test_graph_compiles_with_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver
    g = _builder().compile(checkpointer=InMemorySaver())
    assert g is not None
