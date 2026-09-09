"""祛魅:不用任何框架,手写最裸的 Agent 循环——看清它就是个带工具清单的 for 循环。
用法:.venv/Scripts/python.exe -m scripts.bare_agent_loop "订单1001的物流到哪了"
"""
import asyncio
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm import get_chat_model
from app.core.prompts import AGENT_SYSTEM
from app.tools.infra import execute_tool_call
from app.tools.registry import get_all_tools


async def run_agent(query: str, max_turns: int = 6):
    model = get_chat_model().bind_tools(get_all_tools())
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(query)]
    for step in range(1, max_turns + 1):
        ai = await model.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            print(f"[第{step}步] 无工具调用 → 收敛")
            return ai.content
        for tc in ai.tool_calls:
            print(f"[第{step}步] 调用工具 {tc['name']} args={tc['args']}")
            run = await execute_tool_call(tc, conversation_id=0)
            messages.append(run.tool_message)
    print(f"[封顶] max_turns={max_turns} 用尽仍未收敛,生产环境此处应走兜底话术")
    return "(未收敛)"


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "订单1001的物流到哪了"
    print("问题:", query)
    print("答复:", await run_agent(query))


if __name__ == "__main__":
    asyncio.run(main())
