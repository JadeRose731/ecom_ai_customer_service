.PHONY: dev test eval seed eval-agent

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
