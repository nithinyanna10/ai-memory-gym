.PHONY: install test run_ui run_api

install:
	pip install -r requirements.txt

test:
	cd "$(CURDIR)" && PYTHONPATH=. python -m pytest tests/ -v

run_ui:
	PYTHONPATH=. streamlit run app/ui/streamlit_app.py --server.port 8501

run_api:
	PYTHONPATH=. uvicorn app.api.main:app --host 0.0.0.0 --port 8000
