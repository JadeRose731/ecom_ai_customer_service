"""直连 SiliconFlow /rerank 冒烟。不通即红线停。"""
import httpx

from app.config import settings


def main() -> None:
    url = settings.rerank_base_url.rstrip("/") + "/rerank"
    payload = {
        "model": settings.rerank_model,
        "query": "退货运费谁承担",
        "documents": [
            "满99元包邮,不满收取10元运费。",
            "七天无理由退货,非质量问题退货运费由买家承担。",
            "智能猫砂盆 Pro 型号支持自动清理。",
        ],
        "top_n": 3,
    }
    r = httpx.post(url, json=payload, headers={"Authorization": f"Bearer {settings.rerank_api_key}"}, timeout=60)
    r.raise_for_status()
    data = r.json()
    print("原始返回:", data)
    results = data["results"]
    assert results, "rerank 返回空"
    top = max(results, key=lambda x: x["relevance_score"])
    print(f"最相关 index={top['index']} score={top['relevance_score']:.4f}")
    assert top["index"] == 1, "退货运费问题应命中第 2 条(买家承担)"
    print("GO: rerank 链路通")


if __name__ == "__main__":
    main()
