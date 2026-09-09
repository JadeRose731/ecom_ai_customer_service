.PHONY: dev test eval seed eval-agent kb-preview kb-build kb-vectorize kb-mine kb-reset eval-retrieval seed-conv eval-mining

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
