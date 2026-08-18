UV ?= uv
ANALYSIS_DIR := build/analysis
PAPER_DIR := build/paper
TEX_BIN := $(shell dirname "$$(readlink "$$(command -v pdflatex)" 2>/dev/null || command -v pdflatex)")

.PHONY: analysis build check ci ci-docker clean format lint package paper test

analysis:
	$(UV) run --extra benchmark python self_redaction.py --presidio --output-dir $(ANALYSIS_DIR)

paper: analysis
	mkdir -p $(PAPER_DIR)
	PATH=$(TEX_BIN):$(PATH) TEXINPUTS=$(ANALYSIS_DIR):paper: latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=$(PAPER_DIR) paper/main.tex
	! rg -n "undefined citations|There were undefined citations|Reference .* undefined|Overfull \\\\hbox" $(PAPER_DIR)/main.log

build: paper

package:
	$(UV) build
	$(UV) run --group release twine check dist/*

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

lint:
	$(UV) run ruff format --check .
	$(UV) run ruff check .

test:
	$(UV) run --extra benchmark pytest

check: lint test analysis paper

ci: check

ci-docker:
	docker run --rm -v "$(CURDIR):/work" -v /work/.venv -w /work python:3.13-slim sh -c "pip install uv==0.12.5 && uv sync --locked --all-extras --all-groups && uv run ruff format --check . && uv run ruff check . && uv run --extra benchmark pytest && uv run --extra benchmark python self_redaction.py --presidio --output-dir build/analysis"

clean:
	$(UV) run python -c "import shutil; shutil.rmtree('build', ignore_errors=True); shutil.rmtree('dist', ignore_errors=True)"
