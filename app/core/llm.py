from langchain_openai import ChatOpenAI

from app.config import settings

def get_chat_model(streaming: bool = False, temperature: float = 0.3,
                   model: str | None = None) -> ChatOpenAI:
    """直连聊天上游。地址、模型名、密钥全从 .env 来,换上游不动这里。
    temperature 默认 0.3;裁判类调用(评估两裁判 / judge-check)显式传 0 去抖——
    同一份输入 0.3 下重放会摇摆,0 才能当尺子。
    model 显式覆盖模型名(意图识别可配更大模型求准),None 时回落 settings.chat_model。
    ch09:usage 统一收口,流式也回传 token,Langfuse 按意图统计才不漏账。"""
    return ChatOpenAI(
        model=model or settings.chat_model,
        base_url=settings.chat_base_url,
        api_key=settings.chat_api_key,
        streaming=streaming,
        stream_usage=True,   # ch09:流式也回传 usage(等效 stream_options.include_usage)
        temperature=temperature,
    )
