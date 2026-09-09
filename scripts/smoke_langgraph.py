# scripts/smoke_langgraph.py
"""ch05 红线冒烟:StateGraph + AsyncSqliteSaver + 多模式流式 是否在当前版本跑通,并打印流式形状。

默认走真实上游(get_chat_model);无 key 阶段加 FAKE_MODEL=1 用 langchain-core 的
GenericFakeChatModel——红线要验的是图/持久化/流式的接口形状,不是上游,真上游冒烟待 key。
钉死形状(装的是 langgraph 1.2.11):
  astream(stream_mode=["messages","updates"]) 产出 (mode, chunk) 元组:
    messages 模式 chunk=(msg, metadata),metadata["langgraph_node"] 是产出 token 的节点名;
    updates 模式 chunk={node_name: state_delta}。
"""
import asyncio
import os
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from app.core.llm import get_chat_model


class S(TypedDict):
    messages: Annotated[list, add_messages]


def _model():
    if os.environ.get("FAKE_MODEL"):
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        return GenericFakeChatModel(messages=iter([
            __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                content="我是喵喵优选的客服小猫,很高兴为您服务。")]))
    return get_chat_model(streaming=True)


async def call_model(state: S):
    ai = await _model().ainvoke(state["messages"])
    return {"messages": [ai]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("call_model", call_model)
        b.add_edge(START, "call_model")
        b.add_edge("call_model", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-1"}}
        seen_modes = set()
        async for mode, chunk in graph.astream(
            {"messages": [HumanMessage("用一句话介绍你自己")]},
            config, stream_mode=["messages", "updates"],
        ):
            seen_modes.add(mode)
            if mode == "messages":
                msg, meta = chunk
                print("MSG node=", meta.get("langgraph_node"), "text=", (msg.content or "")[:20])
            else:
                print("UPD", list(chunk.keys()))
        print("OK modes=", seen_modes)


if __name__ == "__main__":
    asyncio.run(main())
