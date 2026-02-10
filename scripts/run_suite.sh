#!/usr/bin/env bash
# Run benchmark suite headlessly: policies x scenarios x seeds.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH}"
python -c "
from bench.suite_runner import run_suite
from bench.schemas import SuiteRunConfig
config = SuiteRunConfig(
    policies=['full_log', 'vector_rag'],
    scenarios=['personal_assistant', 'research', 'ops'],
    seeds=[42, 43],
    number_of_days=5,
    stress_modes=[None],
    llm_mode='mock',
)
result = run_suite(config, out_dir='data/runs', use_cache=True)
print('Suite run_id:', result.run_id)
print('Aggregated CSV:', result.aggregated_csv_path)
print('Manifest:', result.manifest_path)
print('Total runs:', len(result.results))
"
