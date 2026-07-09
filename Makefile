.PHONY: setup test lint fixtures eda up down logs ps clean

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

setup:  ## Create venv and install package + dev deps
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[api,notebook,dev]"

test:  ## Run the pytest suite (generates fixtures automatically)
	$(PY) -m pytest

lint:
	$(VENV)/bin/ruff check src tests

fixtures:  ## Generate synthetic real/fake fixtures under data/interim/fixtures
	$(PY) scripts/make_fixtures.py --out data/interim/fixtures --n 40

eda:  ## Launch the EDA notebook
	$(VENV)/bin/jupyter notebook notebooks/01_eda.ipynb

up:  ## Start the full stack
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.egg-info src/*.egg-info
