"""ch06 红线冒烟:interrupt/Command(resume) 在当前 langgraph 上的中断 surface 形状。

需要打上游吗?不需要——只用一个纯 interrupt 节点。
用法:PYTHONPATH=. uv run python -m scripts.smoke_interrupt
(等价:make smoke-interrupt)

在 langgraph 1.2.11 + langgraph-checkpoint-sqlite 3.1.1 上钉死六段 surface,
供 Task 7(单测断言)与 Task 13(runtime)直接引用:
  A   非流式 ainvoke 遇中断:返回 dict 的 __interrupt__ 结构与取值路径
  B   非流式 Command(resume=v) 续跑
  C   astream(stream_mode=["messages","updates"]) 下中断出现在哪种 chunk
  C'  aget_state 探 pending(interrupts 挂在哪个字段)
  C'' 流式 resume
  D   图外直接调用节点:抛什么、载荷取法(Task 7 单测断言用)
      D1 裸调(无 runnable 上下文)  D2 假 __pregel_scratchpad 最小上下文
"""

import asyncio
from itertools import count
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt


class S(TypedDict):
    messages: Annotated[list, add_messages]
    picked: str


async def ask_order(state: S):
    picked = interrupt({"type": "select_order", "orders": [{"order_id": "1001"}, {"order_id": "2002"}]})
    return {"picked": picked}


async def confirm(state: S):
    return {"messages": [("ai", f"已选订单 {state['picked']}")]}


async def main():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as cp:
        await cp.setup()
        b = StateGraph(S)
        b.add_node("ask_order", ask_order)
        b.add_node("confirm", confirm)
        b.add_edge(START, "ask_order")
        b.add_edge("ask_order", "confirm")
        b.add_edge("confirm", END)
        graph = b.compile(checkpointer=cp)
        config = {"configurable": {"thread_id": "smoke-int-1"}}

        # A) 非流式:首跑应带 __interrupt__
        out = await graph.ainvoke({"messages": [("human", "我要退款")]}, config)
        print("A ainvoke keys=", list(out.keys()))
        print("A __interrupt__=", out.get("__interrupt__"))
        inter = out["__interrupt__"][0]
        print("A 取值路径: out['__interrupt__'][0].value =", inter.value, "| .id =", inter.id)
        assert "__interrupt__" in out, "A: ainvoke 首跑返回缺 __interrupt__"

        # B) 非流式 resume
        out2 = await graph.ainvoke(Command(resume="1001"), config)
        print("B resume picked=", out2.get("picked"), "msgs=", [m.content for m in out2["messages"]])
        print("B resume 完成后 keys=", list(out2.keys()))
        assert out2.get("picked") == "1001", "B: resume 后 picked != 1001"

        # C) 流式:中断信息出现在哪种 chunk
        config2 = {"configurable": {"thread_id": "smoke-int-2"}}
        print("C astream chunks:")
        upd_interrupt_seen = False
        async for mode, chunk in graph.astream(
            {"messages": [("human", "我要退款")]}, config2, stream_mode=["messages", "updates"]
        ):
            if mode == "updates":
                print("  UPD keys=", list(chunk.keys()), "val=", chunk)
                if "__interrupt__" in chunk:
                    upd_interrupt_seen = True
            else:
                # messages 模式:chunk 是 (message_chunk, metadata)
                print("  MSG chunk type=", type(chunk[0]).__name__, "node=", chunk[1].get("langgraph_node"))
        print("C updates chunk 是否出现 __interrupt__ 键:", upd_interrupt_seen)

        # C') 流式后探 pending(备用探测路径)
        snap = await graph.aget_state(config2)
        print(
            "C' aget_state .next=", snap.next,
            "interrupts=", getattr(snap, "interrupts", None),
            "tasks_interrupts=", [t.interrupts for t in snap.tasks],
        )
        if snap.tasks and snap.tasks[0].interrupts:
            print("C' 取值路径: snap.tasks[0].interrupts[0].value =", snap.tasks[0].interrupts[0].value)

        # C'') 流式 resume
        print("C'' astream resume:")
        async for mode, chunk in graph.astream(
            Command(resume="2002"), config2, stream_mode=["messages", "updates"]
        ):
            if mode == "updates":
                print("  ", mode, chunk)
            else:
                print("  ", mode, (chunk[0].content, chunk[1].get("langgraph_node")))

        # D) 图外直接调用被中断节点(无 resume 上下文)——Task 7 fetch_order 单测要断言这个形状
        await section_d()

    print("\nSMOKE OK:中断 surface 形状已钉死(见上方 A/B/C/C'/C''/D 各段输出)")


async def section_d():
    from langgraph.errors import GraphInterrupt

    # D1) 裸调:不走任何 runnable 上下文,直接 await 节点函数
    print("D1 裸调 ask_order(无 runnable 上下文):")
    try:
        await ask_order({"messages": []})
        raise AssertionError("D1: interrupt() 未抛异常,中断 surface 与预期不符")
    except GraphInterrupt as e:
        print("D1 GraphInterrupt args=", e.args, "first.value=", e.args[0][0].value if e.args else None)
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001 —— 记录真实异常类型,这正是冒烟要钉死的
        print(f"D1 实际抛 {type(e).__module__}.{type(e).__name__}: {e}")

    # D2) 带最小 runnable 上下文(假 __pregel_scratchpad)——Task 7 纯单测可直接复制的取法
    from langchain_core.runnables.config import var_child_runnable_config
    from langgraph._internal._scratchpad import PregelScratchpad

    fake_cfg = {
        "configurable": {
            "thread_id": "smoke-d",
            "checkpoint_ns": "",
            "__pregel_scratchpad": PregelScratchpad(
                step=0,
                stop=1,
                call_counter=count(1).__next__,
                interrupt_counter=count().__next__,
                get_null_resume=lambda _consume: None,
                resume=[],
                subgraph_counter=count(1).__next__,
            ),
        }
    }
    print("D2 带最小 runnable 上下文(假 __pregel_scratchpad)再调:")
    var_child_runnable_config.set(fake_cfg)
    try:
        await ask_order({"messages": []})
        raise AssertionError("D2: interrupt() 未抛 GraphInterrupt,单测取法不成立")
    except GraphInterrupt as e:
        print("D2 GraphInterrupt args=", e.args)
        print("D2 取值路径: e.args[0][0].value =", e.args[0][0].value, "| e.args[0][0].id =", e.args[0][0].id)
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"D2 实际抛 {type(e).__module__}.{type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
