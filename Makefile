.PHONY: venv install test run_ui run_api run_suite

# Create .venv and install deps (Python 3.11+)
venv:
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"
	@.venv/bin/pip install -r requirements.txt

install:
	pip install -r requirements.txt

test:
	cd "$(CURDIR)" && PYTHONPATH=. python -m pytest tests/ -v

run_ui:
	PYTHONPATH=. streamlit run app/ui/streamlit_app.py --server.port 8501

run_api:
	PYTHONPATH=. uvicorn app.api.main:app --host 0.0.0.0 --port 8000

run_suite:
	PYTHONPATH=. python -c "\
from bench.suite_runner import run_suite;\
from bench.schemas import SuiteRunConfig;\
r = run_suite(SuiteRunConfig(policies=['full_log','vector_rag'], scenarios=['personal_assistant','ops'], seeds=[42,43], number_of_days=5), use_cache=True);\
print('Suite run_id:', r.run_id, '| CSV:', r.aggregated_csv_path)"
