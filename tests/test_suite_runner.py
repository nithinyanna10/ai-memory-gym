"""Tests for suite runner output and caching."""

import json
import os
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.suite_runner import run_suite, _config_hash
from bench.schemas import SuiteRunConfig, BenchmarkConfig


def test_config_hash_deterministic():
    c1 = BenchmarkConfig(scenario_type="a", policy="b", seed=42)
    c2 = BenchmarkConfig(scenario_type="a", policy="b", seed=42)
    assert _config_hash(c1) == _config_hash(c2)
    c3 = BenchmarkConfig(scenario_type="a", policy="c", seed=42)
    assert _config_hash(c1) != _config_hash(c3)


def test_suite_runner_produces_files():
    with tempfile.TemporaryDirectory() as tmp:
        config = SuiteRunConfig(
            policies=["full_log"],
            scenarios=["personal_assistant"],
            seeds=[42],
            number_of_days=3,
            stress_modes=[None],
            llm_mode="mock",
        )
        result = run_suite(config, out_dir=tmp, use_cache=False)
        assert result.run_id
        assert result.aggregated_csv_path and os.path.exists(result.aggregated_csv_path)
        assert result.manifest_path and os.path.exists(result.manifest_path)
        assert len(result.results) >= 1
        with open(result.manifest_path) as f:
            manifest = json.load(f)
        assert manifest["suite_run_id"] == result.run_id
        assert "n_runs" in manifest
