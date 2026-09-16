from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 这三项故意不给默认值,缺了在启动就报 Field required,直接指向 .env
    chat_model: str
    chat_base_url: str
    chat_api_key: str
    token_budget: int = 2000
    # 本机 3306 被 Windows 版 MySQL 占用,Docker MySQL 映射到 3307
    database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp"
    test_database_url: str = "mysql+asyncmy://root:root@localhost:3307/mewhelp_test"
    # ch03 嵌入与向量检索
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_api_key: str                      # 跟着账号走,不给默认值
    embed_model: str = "BAAI/bge-m3"        # 上游真实名,没有别名这一层
    milvus_uri: str = "http://localhost:19530"
    retrieval_top_k: int = 3
    retrieval_min_score: float = 0.4
    # ch04 重排:bge-reranker-v2-m3 走 Jina / Cohere 那套 /rerank 形状,不是 OpenAI 协议
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str                      # 跟着账号走,不给默认值
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    # ch04 混合检索 + 重排:召回两路各取 Top-50 → 精排 Top-10 → 最高分低于阈值判证据低
    recall_top_k: int = 50
    rerank_top_k: int = 10
    rerank_min_score: float = 0.3
    # ch05 图编排
    max_agent_steps: int = 6              # ReAct 环最大步数(封顶,超则兜底)
    # token 花销不在环里卡:那是成本控制,归 ch09 跟 Langfuse 的账一起看
    checkpointer_db_path: str = "data/ch05_checkpoints.sqlite"  # LangGraph checkpointer(data/*.db* 已 gitignore)
    # ch06 意图识别模型配置(先求准:默认大模型;降级路仅种子,本章不实现运行时)
    intent_model: str = ""            # 意图识别模型名;空=回落 chat_model
    intent_small_model: str = ""      # 降级路小模型(种子,未接运行时)
    intent_mode: str = "accuracy"     # accuracy=只用大模型;cost=小模型判→低置信升级大模型(未实现)
    intent_conf_threshold: float = 0.6  # cost 模式升级阈值(种子)

settings = Settings()
