"""Load and list runs from disk (data/runs) for UI."""

import json
from pathlib import Path
from typing import Optional, Any

# Load from bench when needed to avoid circular imports
def _load_result(path: str):
    from bench.runner import load_result
    return load_result(path)


def leaderboard_rows_from_disk(data_dir: str) -> list[dict[str, Any]]:
    """Read manifests from data/runs/runs/*/manifest.json and return leaderboard rows."""
    rows = []
    data_dir = Path(data_dir)
    runs_dir = data_dir / "runs"
    if not runs_dir.exists():
        return rows
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        m_path = d / "manifest.json"
        if not m_path.exists():
            continue
        try:
            with open(m_path) as fp:
                man = json.load(fp)
            rows.append({
                "run_id": man.get("run_id", d.name),
                "scenario": man.get("scenario_type", ""),
                "policy": man.get("policy", ""),
                "stress": man.get("stress_mode") or "-",
                "seed": man.get("seed"),
                "accuracy": man.get("accuracy", 0),
                "m_score": man.get("m_score"),
                "tokens": man.get("token_estimate", 0),
            })
        except Exception:
            pass
    return rows


def list_run_ids_from_disk(data_dir: str) -> list[tuple[str, Optional[str]]]:
    """Return list of (run_id, path_to_json). path is None if only manifest exists (no full result)."""
    data_dir = Path(data_dir)
    run_ids = {}
    for f in data_dir.glob("run_*.json"):
        run_id = f.stem.replace("run_", "", 1)
        run_ids[run_id] = str(f)
    runs_dir = data_dir / "runs"
    if runs_dir.exists():
        for d in runs_dir.iterdir():
            if d.is_dir():
                m = d / "manifest.json"
                if m.exists():
                    try:
                        with open(m) as fp:
                            man = json.load(fp)
                        rid = man.get("run_id")
                        if rid and rid not in run_ids:
                            p = data_dir / f"run_{rid}.json"
                            run_ids[rid] = str(p) if p.exists() else None
                    except Exception:
                        pass
    return list(run_ids.items())


def load_result_from_disk(data_dir: str, run_id: str):
    """Load BenchmarkResult from data_dir/run_<run_id>.json. Returns None if file missing."""
    path = Path(data_dir) / f"run_{run_id}.json"
    if not path.exists():
        return None
    return _load_result(str(path))
