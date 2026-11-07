"""Dated, SHA-pinned snapshots of the Greco1899/scrape_ufc_stats CSVs.

Data from Greco1899/scrape_ufc_stats (GPL-3.0), originally scraped from ufcstats.com.

Each snapshot is downloaded from the raw URL *at a specific upstream commit*, so
the committed manifest alone is enough to re-download identical files later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ufc_model.config import Config
from ufc_model.http import PoliteClient

log = logging.getLogger(__name__)

SOURCE_CREDIT = (
    "Data from Greco1899/scrape_ufc_stats (GPL-3.0), originally scraped from ufcstats.com."
)


@dataclass
class UpstreamCommit:
    sha: str
    committed_at: str
    message: str


@dataclass
class SnapshotManifest:
    upstream_repo: str
    upstream_sha: str
    upstream_committed_at: str
    upstream_message: str
    retrieved_at: str
    files: dict[str, dict]
    source_credit: str = SOURCE_CREDIT


def latest_upstream_commit(client: PoliteClient, cfg: Config) -> UpstreamCommit:
    """Look up the head commit of the upstream branch via the GitHub API."""
    url = f"https://api.github.com/repos/{cfg.greco.repo}/commits/{cfg.greco.branch}"
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    data = client.get(url, headers=headers).json()
    return UpstreamCommit(
        sha=data["sha"],
        committed_at=data["commit"]["committer"]["date"],
        message=data["commit"]["message"].splitlines()[0],
    )


def snapshot_dir(cfg: Config, commit: UpstreamCommit) -> Path:
    day = commit.committed_at[:10]
    return cfg.paths.raw_dir / "greco" / f"{day}_{commit.sha[:10]}"


def download_snapshot(client: PoliteClient, cfg: Config, commit: UpstreamCommit) -> Path:
    """Download every CSV at ``commit`` unless that snapshot already exists locally."""
    out = snapshot_dir(cfg, commit)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        log.info("Snapshot %s already present; not re-downloading", out.name)
        return out
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    for name in cfg.greco.files:
        url = f"https://raw.githubusercontent.com/{cfg.greco.repo}/{commit.sha}/{name}"
        content = client.get(url).content
        if not content.strip():
            raise ValueError(f"Upstream file {name} at {commit.sha} is empty")
        (out / name).write_bytes(content)
        files[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "lines": content.count(b"\n"),
        }
        log.info("Downloaded %s (%d bytes)", name, len(content))
    manifest = SnapshotManifest(
        upstream_repo=cfg.greco.repo,
        upstream_sha=commit.sha,
        upstream_committed_at=commit.committed_at,
        upstream_message=commit.message,
        retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        files=files,
    )
    text = json.dumps(asdict(manifest), indent=2)
    manifest_path.write_text(text)
    committed = cfg.paths.manifests_dir / "greco" / f"{out.name}.json"
    committed.parent.mkdir(parents=True, exist_ok=True)
    committed.write_text(text)
    return out


def read_manifest(snapshot: Path) -> SnapshotManifest:
    return SnapshotManifest(**json.loads((snapshot / "manifest.json").read_text()))


def verify_snapshot(snapshot: Path) -> None:
    """Raise if any file no longer matches the checksum recorded in its manifest."""
    manifest = read_manifest(snapshot)
    for name, meta in manifest.files.items():
        digest = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if digest != meta["sha256"]:
            raise ValueError(f"Checksum mismatch for {snapshot / name}")


def latest_local_snapshot(cfg: Config) -> Path:
    """Most recent complete local snapshot (for offline runs)."""
    root = cfg.paths.raw_dir / "greco"
    snaps = sorted(p for p in root.glob("*") if (p / "manifest.json").exists())
    if not snaps:
        raise FileNotFoundError(f"No local snapshots under {root}")
    return snaps[-1]
