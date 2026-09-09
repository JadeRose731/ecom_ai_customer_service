# app/kb/milvus_client.py
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024


def get_client(uri: str | None = None) -> MilvusClient:
    # Milvus Lite(本地文件)无 Windows 包;经确认跑 Docker Standalone(http uri)。
    return MilvusClient(uri=uri or settings.milvus_uri)


def ensure_collection(client: MilvusClient) -> None:
    """幂等:已存在则确保加载;不存在则按 schema 建集合 + AUTOINDEX/COSINE 索引 + 加载。
    主键 id = MySQL knowledge_chunks.id(auto_id=False,由我方提供),使重跑按 id upsert 幂等。"""
    if client.has_collection(COLLECTION):
        client.load_collection(COLLECTION)
        return
    schema = CollectionSchema([
        FieldSchema("id", DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema("vector", DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema("question", DataType.VARCHAR, max_length=2048),
        FieldSchema("answer", DataType.VARCHAR, max_length=8192),
    ])
    client.create_collection(COLLECTION, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_index(COLLECTION, index_params)
    client.load_collection(COLLECTION)


def upsert_vectors(client: MilvusClient, rows: list[dict]) -> None:
    if rows:
        client.upsert(COLLECTION, rows)


def search(client: MilvusClient, vector: list[float], top_k: int) -> list[dict]:
    res = client.search(
        COLLECTION, data=[vector], limit=top_k,
        output_fields=["question", "answer"],
        search_params={"metric_type": "COSINE"},
    )
    return [
        {"id": h["id"], "score": float(h["distance"]),
         "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
        for h in res[0]
    ]


def count(client: MilvusClient) -> int:
    return client.query(COLLECTION, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]
