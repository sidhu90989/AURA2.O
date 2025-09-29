SHELL := /bin/bash

.PHONY: help setup venv backend-deps frontend-deps dev backend frontend clean

help:
	@echo "Targets:"
	@echo "  setup          Create venv + install backend & frontend deps"
	@echo "  backend        Run FastAPI backend (reload)"
	@echo "  frontend       Run Next.js dev server"
	@echo "  dev            Run backend then frontend (two processes)"
	@echo "  clean          Remove virtual environment and caches"

venv:
	python3 -m venv .venv || true
	. .venv/bin/activate; pip install --upgrade pip wheel setuptools

backend-deps: venv
	. .venv/bin/activate; pip install -r backend/requirements.txt

frontend-deps:
	cd frontend && npm install

setup: backend-deps frontend-deps

backend:
	. .venv/bin/activate; uvicorn backend.app.main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

dev:
	( . .venv/bin/activate; uvicorn backend.app.main:app --reload --port 8000 ) & \
	BACK_PID=$$!; \
	cd frontend && npm run dev; \
	kill $$BACK_PID || true

clean:
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
