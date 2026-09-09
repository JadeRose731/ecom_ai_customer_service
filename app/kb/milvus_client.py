# app/kb/milvus_client.py
# ch04:集合重建为 dense(bge-m3)+ text(中文分词)+ sparse(BM25 Function)+ 标量元数据。
# BM25 全文检索只有 Standalone 支持(无 Lite),本仓库 ch03 起即跑 Docker Standalone。
from pymilvus import (
    AnnSearchRequest, DataType, Function, FunctionType, MilvusClient, RRFRanker,
)

from app.config import settings

COLLECTION = "knowledge"
DIM = 1024

_CLIENT: MilvusClient | None = None
_ensured: set[str] = set()

_OUTPUT = ["question", "answer", "section_path", "content_type", "category"]


def get_client(uri: str | None = None) -> MilvusClient:
    """默认返回进程内单例(连 Standalone);传 uri(测试)返回独立客户端。"""
    global _CLIENT
    if uri is not None:
        return MilvusClient(uri=uri)
    if _CLIENT is None:
        _CLIENT = MilvusClient(uri=settings.milvus_uri)
    return _CLIENT


def ensure_collection(client: MilvusClient, collection: str = COLLECTION) -> None:
    """幂等建集合:dense(COSINE) + text(chinese analyzer) + sparse(BM25 Function) + 标量字段。
    单例默认集合只实际执行一次;不同集合名(测试临时库)各自 ensure。
    注意:ch03 旧 schema 的同名集合不带本 schema,升级需 drop 重建(make kb-reset)。"""
    if client is _CLIENT and collection in _ensured:
        return
    if client.has_collection(collection):
        client.load_collection(collection)
    else:
        schema = client.create_schema(auto_id=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=DIM)
        schema.add_field("text", DataType.VARCHAR, max_length=8192,
                         enable_analyzer=True, analyzer_params={"type": "chinese"})
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("question", DataType.VARCHAR, max_length=2048)
        schema.add_field("answer", DataType.VARCHAR, max_length=8192)
        schema.add_field("section_path", DataType.VARCHAR, max_length=512)
        schema.add_field("content_type", DataType.VARCHAR, max_length=32)
        schema.add_field("category", DataType.VARCHAR, max_length=255)
        # BM25 Function 必须在建集合前挂上:写 text 时系统自动产出 sparse 向量
        schema.add_function(Function(
            name="text_bm25", input_field_names=["text"],
            output_field_names=["sparse"], function_type=FunctionType.BM25,
        ))
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        client.create_collection(collection, schema=schema, index_params=index_params)
        client.load_collection(collection)
    if client is _CLIENT:
        _ensured.add(collection)


def upsert_vectors(client: MilvusClient, rows: list[dict], collection: str = COLLECTION) -> None:
    if rows:
        client.upsert(collection, rows)


def flush(client: MilvusClient, collection: str = COLLECTION) -> None:
    """刷盘:Standalone Bounded 一致性下,新 upsert 的数据须 flush 后才可被 BM25/检索命中(ch03 实证)。"""
    client.flush(collection)


def _cat_expr(category: str | None) -> str:
    return f'category == "{category}"' if category else ""


def _hit(h: dict) -> dict:
    e = h["entity"]
    return {"id": h["id"], "score": float(h["distance"]),
            "question": e["question"], "answer": e["answer"],
            "section_path": e["section_path"], "content_type": e["content_type"],
            "category": e["category"]}


def dense_search(client, vector, top_k, category=None, collection=COLLECTION) -> list[dict]:
    res = client.search(collection, data=[vector], anns_field="dense", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "COSINE"},
                        filter=_cat_expr(category))
    # Standalone COSINE 的 distance 即相似度(越大越相似)
    return [_hit(h) for h in res[0]]


def bm25_search(client, text, top_k, category=None, collection=COLLECTION) -> list[dict]:
    res = client.search(collection, data=[text], anns_field="sparse", limit=top_k,
                        output_fields=_OUTPUT, search_params={"metric_type": "BM25"},
                        filter=_cat_expr(category))
    return [_hit(h) for h in res[0]]


def hybrid_search(client, vector, text, top_k, recall=50, category=None, collection=COLLECTION) -> list[dict]:
    expr = _cat_expr(category)
    dense_req = AnnSearchRequest(data=[vector], anns_field="dense",
                                 param={"metric_type": "COSINE"}, limit=recall, expr=expr)
    sparse_req = AnnSearchRequest(data=[text], anns_field="sparse",
                                  param={"metric_type": "BM25"}, limit=recall, expr=expr)
    res = client.hybrid_search(collection, reqs=[dense_req, sparse_req], ranker=RRFRanker(),
                               limit=top_k, output_fields=_OUTPUT)
    return [_hit(h) for h in res[0]]


def search(client: MilvusClient, vector: list[float], top_k: int) -> list[dict]:
    """ch03 兼容口(dense 单路,只回 question/answer)。ch04 起新代码用 dense_search。"""
    res = client.search(COLLECTION, data=[vector], anns_field="dense", limit=top_k,
                        output_fields=["question", "answer"],
                        search_params={"metric_type": "COSINE"})
    return [{"id": h["id"], "score": float(h["distance"]),
             "question": h["entity"]["question"], "answer": h["entity"]["answer"]}
            for h in res[0]]


def count(client, collection: str = COLLECTION) -> int:
    return client.query(collection, filter="id >= 0", output_fields=["count(*)"])[0]["count(*)"]


def drop(client, collection: str) -> None:
    if client.has_collection(collection):
        client.drop_collection(collection)
    if client is _CLIENT:
        _ensured.discard(collection)
