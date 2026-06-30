# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python (3.12) project. The **core runs with zero third-party dependencies** (standard library only); `openai` (in `requirements.txt`) is optional and only enables real-LLM mode.

- A virtualenv is provided at `.venv` (the startup update script keeps `openai` installed there). Use `.venv/bin/python` to run anything that needs `openai`; plain `python3` is enough for the stdlib-only paths.
- **No LLM key is required to run or test.** Without `LLM_API_KEY` the agent automatically falls back to the deterministic `HeuristicInspector`, so all scripts and the full test suite pass offline. To enable real LLM mode, copy `.env.example` to `.env` and set `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (any OpenAI-compatible endpoint).
- **There is no configured linter** (no flake8/ruff/black config). Lint is not part of this repo's workflow.
- Tests: `.venv/bin/python -m unittest tests.test_qc_agent -v` (runs fully offline).
- Run/CLI entry points (see `README.md` §4 for full usage):
  - Single call: `.venv/bin/python scripts/inspect_text.py "left:...text..."` (add `--tools` for the agentic tool loop).
  - Batch eval: `.venv/bin/python scripts/batch_eval.py --csv data/sample_cases.csv --out results.csv`.
  - Label governance: `.venv/bin/python scripts/evolve.py --csv data/sample_cases.csv --out conflicts.csv`.
- Outputs `results.csv` / `conflicts.csv` and `.env` / `.cache/` are gitignored.
