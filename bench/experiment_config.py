"""Production-grade ExperimentConfig (pydantic) and RunManifest with stable hashing."""

import hashlib
import json
from typing import Optional, Any

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Reproducible experiment configuration. Same config + seed => same outputs."""

    scenario_type: str = Field(default="personal_assistant", description="Scenario pack name")
    policy: str = Field(default="full_log", description="Memory policy")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    number_of_days: int = Field(default=7, ge=1, le=90)
    wm_size: int = Field(default=10, ge=1, le=100)
    top_k: int = Field(default=5, ge=1, le=50)
    decay_lambda: float = Field(default=0.1, ge=0.0, le=1.0)
    salience_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    rehearsal_frequency: int = Field(default=3, ge=1, le=30)
    llm_mode: str = Field(default="mock", description="mock | rule | real")
    stress_mode: Optional[str] = Field(default=None, description="None | distraction_flood | memory_corruption | contradiction_injection | distribution_shift")
    stress_kwargs: dict = Field(default_factory=dict)

    def to_benchmark_config(self):
        """Convert to legacy BenchmarkConfig for runner."""
        from bench.schemas import BenchmarkConfig
        return BenchmarkConfig(
            scenario_type=self.scenario_type,
            policy=self.policy,
            seed=self.seed,
            number_of_days=self.number_of_days,
            wm_size=self.wm_size,
            top_k=self.top_k,
            decay_lambda=self.decay_lambda,
            salience_threshold=self.salience_threshold,
            rehearsal_frequency=self.rehearsal_frequency,
            use_mock_llm=(self.llm_mode != "real"),
            llm_mode=self.llm_mode,
            stress_mode=self.stress_mode,
            stress_kwargs=self.stress_kwargs,
        )


class RunManifest(BaseModel):
    """Manifest for a single run: config hash, run_id, paths, summary metrics."""

    run_id: str
    config_hash: str
    config: dict = Field(default_factory=dict)
    scenario_type: str = ""
    policy: str = ""
    seed: Optional[int] = None
    stress_mode: Optional[str] = None
    accuracy: float = 0.0
    m_score: Optional[float] = None
    token_estimate: int = 0
    memory_items_stored: int = 0
    cached: bool = False
    artifacts: dict = Field(default_factory=dict, description="Paths: manifest, traces, metrics, summary_csv, summary_parquet, run_log")


def stable_config_hash(config: Any) -> str:
    """Stable hash from config (sorted keys). Same config => same hash."""
    if hasattr(config, "model_dump"):
        d = config.model_dump()
    elif hasattr(config, "__dict__"):
        d = {k: v for k, v in config.__dict__.items() if not k.startswith("_")}
    else:
        d = dict(config) if hasattr(config, "items") else {}
    # Normalize for JSON
    def norm(o):
        if isinstance(o, dict):
            return {str(k): norm(v) for k, v in sorted(o.items())}
        if isinstance(o, list):
            return [norm(x) for x in o]
        return o
    payload = json.dumps(norm(d), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
