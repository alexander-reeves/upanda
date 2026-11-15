PYTHON ?= python
.PHONY: help test tests test-all test-reference coverage style-check reformat docs docs-serve build sdist
help:
	@echo 'test/tests, test-all, test-reference, coverage, style-check, reformat, docs, docs-serve, build, sdist'
test tests:
	$(PYTHON) -m pytest -m 'not reference' tests
test-all:
	$(PYTHON) -m tox
test-reference:
	$(PYTHON) -m pytest -m reference tests/test_reference.py
coverage:
	$(PYTHON) -m pytest --cov=UPanda --cov-report=term-missing --cov-report=html --cov-report=xml
style-check:
	$(PYTHON) -m ruff check src/UPanda tests scripts
reformat:
	$(PYTHON) -m black src/UPanda tests scripts
docs:
	$(PYTHON) -m mkdocs build --strict
docs-serve:
	$(PYTHON) -m mkdocs serve
build:
	$(PYTHON) -m build
sdist:
	$(PYTHON) -m build --sdist
