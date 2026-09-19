.PHONY: mcp-up mcp-down dev test eval seed eval-agent kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval seed-conv eval-mining milvus-up milvus-down smoke-rag eval-rag eval-check dev-vectors judge-check eval-rewrite eval-ch05 smoke-interrupt eval-ch06 eval-ch07 langfuse-up langfuse-down calibrate-confidence flywheel flywheel-samples eval-flywheel cost-report

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

# ch04:裁判一致性回归(台账已处置个案原样重放;不一致退出码 1,需真实上游)
judge-check:
	PYTHONPATH=. uv run python scripts/judge_check.py

# ch05:五验收端到端评估(需全服务起 + 真实上游;走 /api/agent 看工具轨迹与建议动作)
eval-ch05:
	PYTHONPATH=. uv run python scripts/eval_ch05.py

# ch06:interrupt/resume 中断 surface 红线冒烟(纯 interrupt 节点,不需要上游)
smoke-interrupt:
	PYTHONPATH=. uv run python scripts/smoke_interrupt.py

# ch06:四验收端到端评估(需全服务起 + 已应用 sql/ch06-ticket-type.sql + 真实 key)
eval-ch06:
	PYTHONPATH=. uv run python scripts/eval_ch06.py

# ch07:摘要 prompt 标注样例验证(需真实上游;structured output,不打本地服务)
eval-ch07:
	PYTHONPATH=. uv run python -m scripts.eval_ch07

# ch08: 两台业务 MCP Server 起停(独立进程,Streamable HTTP :8101/:8102)
mcp-up:
	@mkdir -p log data
	@nohup uv run python mcp_servers/logistics_server.py  > log/mcp-logistics.log 2>&1 & echo $$! > data/mcp-logistics.pid
	@nohup uv run python mcp_servers/aftersales_server.py > log/mcp-aftersales.log 2>&1 & echo $$! > data/mcp-aftersales.pid
	@sleep 1 && echo "MCP Servers 已拉起: logistics=:8101 aftersales=:8102(pid 见 data/*.pid)"

mcp-down:
	-@kill `cat data/mcp-logistics.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-logistics.pid
	-@kill `cat data/mcp-aftersales.pid 2>/dev/null` 2>/dev/null; rm -f data/mcp-aftersales.pid
	@echo "MCP Servers 已停"

# ch08:验收样例端到端(需 make dev 全服务 + mcp-up)
eval-ch08:
	PYTHONPATH=. uv run python scripts/eval_ch08.py

# ch09: Langfuse 自部署观测栈(web:3000 + worker + postgres + clickhouse + redis + minio)
langfuse-up:
	docker compose -f docker-compose.langfuse.yml up -d
	@echo "Langfuse 起中: http://localhost:3000 (admin@mewhelp.local / mewhelp123)"
	@echo "首次就绪约 2-3 分钟;key 已 headless 预置,写 .env:"
	@echo "  LANGFUSE_PUBLIC_KEY=pk-lf-mewhelp-local"
	@echo "  LANGFUSE_SECRET_KEY=sk-lf-mewhelp-local"
	@echo "  LANGFUSE_BASE_URL=http://localhost:3000"

langfuse-down:
	docker compose -f docker-compose.langfuse.yml down

# ch09:置信度阈值校准(需 milvus + 上游可用 + 知识库已建)
calibrate-confidence:
	PYTHONPATH=. uv run python scripts/calibrate_confidence.py

# ch09:飞轮批处理:问题池 → 标准化查重 → 待审队列(需 mysql + 上游可用)
flywheel:
	PYTHONPATH=. uv run python scripts/flywheel_pipeline.py

# ch09:标准化查重 prompt 标注样例验证(通过率 ≥ 80%,需上游可用)
flywheel-samples:
	PYTHONPATH=. uv run python scripts/validate_flywheel_samples.py

# ch09:评估流水线:复用 ch04 评估集,落 eval_runs 连趋势(TRIGGER=手动|定时,需全上游可用)
eval-flywheel:
	PYTHONPATH=. uv run python scripts/eval_flywheel.py --triggered-by $(or $(TRIGGER),手动)

# ch09:按意图 token 账(需 Langfuse 在跑;DAYS=窗口天数,默认 7)
cost-report:
	PYTHONPATH=. uv run python scripts/cost_by_intent.py --days $(or $(DAYS),7)

# ch10: 主题分类器(数据→训练→评测→ONNX→旁路批量归类)
ch10-golden:  ## 预标 prompt 黄金样例验证(通过率 ≥ 80% 才放行批量预标,需上游可用)
	PYTHONPATH=. uv run python scripts/ch10/validate_golden.py

ch10-corpus: ch10-golden  ## 语料流水线:捞池→清洗→预标→模拟补足→抽审导出(依赖验证闸过线;需上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_corpus.py

ch10-dataset:  ## 分层划分 80/10/10 + 训练集增强(需上游可用)
	PYTHONPATH=. uv run python scripts/ch10/build_dataset.py

ch10-train:  ## RoBERTa-wwm-ext 全参微调(MPS/CUDA/CPU 自适应,重依赖走 ml 组)
	PYTHONPATH=. uv run --group ml python scripts/ch10/train.py
