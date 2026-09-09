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

settings = Settings()
