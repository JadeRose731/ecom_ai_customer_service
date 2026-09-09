from langchain_openai import ChatOpenAI

from app.config import settings

def get_chat_model(streaming: bool = False, temperature: float = 0.3) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。
    默认 0.3;裁判类调用(评估两裁判 / judge-check)显式传 0 去抖——
    同一份输入 0.3 下重放会摇摆,0 才能当尺子。"""
    return ChatOpenAI(
        model=settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        temperature=temperature,
    )
