"""Versioned model registry: ``models/registry/<version>/{model.joblib, metadata.json}``.

Only the newest ``keep`` versions stay on disk (and so in git). Every logged
prediction records its model version, and metadata records the data cutoff,
training date and recent-card metrics.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from ufc_model.config import PROJECT_ROOT
from ufc_model.models.production import ProductionModel


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_version(m: ProductionModel) -> str:
    """Deterministic: same data cutoff + same coefficients -> same version."""
    lr = m.pipeline[-1]
    digest = hashlib.sha256(
        np.concatenate([lr.coef_.ravel(), lr.intercept_]).round(10).tobytes()
        + json.dumps(m.columns).encode()
    ).hexdigest()[:8]
    return f"lr-{m.data_cutoff:%Y%m%d}-{digest}"


def save(m: ProductionModel, root: Path, extra: dict, keep: int) -> tuple[str, Path]:
    version = model_version(m)
    out = root / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, out / "model.joblib")
    meta = {
        "version": version,
        "model_type": "logistic_regression (StandardScaler + L2), uncalibrated",
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_cutoff": str(m.data_cutoff),
        "train_start_year": m.train_start_year,
        "n_train_fights": m.n_train_fights,
        "params": m.params,
        "n_features": len(m.columns),
        "code_version": _git_sha(),
        **extra,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    prune(root, keep)
    return version, out


def versions(root: Path) -> list[Path]:
    """Version directories, oldest first (by recorded training time)."""
    dirs = [d for d in root.glob("*") if (d / "metadata.json").exists()]
    return sorted(dirs, key=lambda d: json.loads((d / "metadata.json").read_text())["trained_at"])


def prune(root: Path, keep: int) -> list[str]:
    removed = []
    for d in versions(root)[:-keep] if keep > 0 else []:
        shutil.rmtree(d)
        removed.append(d.name)
    return removed


def load_latest(root: Path) -> tuple[ProductionModel, dict]:
    vs = versions(root)
    if not vs:
        raise FileNotFoundError(f"No model versions under {root}")
    return joblib.load(vs[-1] / "model.joblib"), json.loads((vs[-1] / "metadata.json").read_text())
