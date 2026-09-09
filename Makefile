.PHONY: dev test eval seed eval-agent kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval seed-conv eval-mining milvus-up milvus-down smoke-rag eval-rag eval-check dev-vectors eval-rewrite

dev:
	./scripts/dev.sh

test:
	uv run pytest -v

eval:
	uv run python scripts/eval_extract.py

seed:
	docker compose exec -T mysql mysql -uroot -proot mewhelp < sql/ch02-seed.sql

eval-agent:
	uv run python scripts/eval_agent.py

kb-preview:
	PYTHONPATH=. uv run python scripts/preview_kb.py

kb-build:
	PYTHONPATH=. uv run python scripts/build_kb.py

kb-vectorize:
	PYTHONPATH=. uv run python scripts/vectorize_kb.py

kb-mine:
	PYTHONPATH=. uv run python scripts/mine_knowledge.py

kb-reset:
	PYTHONPATH=. uv run python scripts/reset_kb.py

eval-retrieval:
	PYTHONPATH=. uv run python scripts/eval_retrieval.py

seed-conv:
	docker exec -i mewhelp-mysql mysql --default-character-set=utf8mb4 -uroot -proot mewhelp < sql/ch03-seed.sql

eval-mining:
	PYTHONPATH=. uv run python scripts/eval_mining.py

# ch04:Milvus 三件套起停。服务名是本仓库 compose 里的(milvus/milvus-etcd/milvus-minio),
# 不是官方文件的 etcd/minio/milvus-standalone——ch03 迁 Standalone 时已按本仓库惯例命名。
milvus-up:
	docker compose up -d milvus milvus-etcd milvus-minio
	@echo "等待 Milvus 就绪(healthz)..."; \
	for i in $$(seq 1 60); do \
	  curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && echo "Milvus OK" && exit 0; \
	  sleep 3; done; echo "Milvus 未就绪" && exit 1

milvus-down:
	docker compose stop milvus milvus-etcd milvus-minio

smoke-rag:
	PYTHONPATH=. uv run python scripts/smoke_milvus_bm25.py
	PYTHONPATH=. uv run python scripts/smoke_rerank.py

eval-rewrite:
	PYTHONPATH=. uv run python scripts/eval_rewrite.py

# ch04:四策略评估三段跑齐(检索/证据覆盖度确定性必出;生成段无 key 时自动跳过)
eval-rag:
	PYTHONPATH=. uv run python scripts/eval_ch04.py

# ch04:评估集守门(五桶各 60 / 小节与要点逐字校准 / D 桶规则),改题后必跑
eval-check:
	PYTHONPATH=. uv run python scripts/validate_eval_ch04.py

# ch04:dev 占位灌库(pytest 每轮清空集合;无 key 阶段重灌真实文本 + 占位向量,BM25 真实)
dev-vectors:
	PYTHONPATH=. uv run python scripts/populate_dev_vectors.py
