"""Typed project configuration loaded from ``config/config.yaml``."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ProjectConfig(BaseModel):
    name: str
    version: str
    repo_url: str
    seed: int

    @property
    def user_agent(self) -> str:
        """Generic, identifiable UA: project name and repo URL, no personal data."""
        return f"{self.name}/{self.version} (+{self.repo_url})"


class PathsConfig(BaseModel):
    raw_dir: Path
    manifests_dir: Path
    db_path: Path
    aliases: Path

    def resolved(self, root: Path) -> PathsConfig:
        """Return a copy with every path made absolute relative to ``root``."""
        return PathsConfig(**{k: (root / v if not v.is_absolute() else v) for k, v in self})


class HttpConfig(BaseModel):
    min_delay_s: float
    max_delay_s: float
    timeout_s: float
    max_retries: int


class GrecoConfig(BaseModel):
    repo: str
    branch: str
    files: list[str]


class WikipediaConfig(BaseModel):
    api_url: str
    events_list_page: str
    upcoming_events_to_fetch: int


class StalenessConfig(BaseModel):
    grace_days: int


class NamesConfig(BaseModel):
    suggestion_min_similarity: float
    weight_margin_lbs: float
    min_age_years: float
    max_age_years: float


class FeaturesConfig(BaseModel):
    train_start_year: int
    recent_fights: int
    shrink_minutes: float
    shrink_sig_attempts: float
    shrink_td_attempts: float
    default_height_in: float
    default_reach_in: float
    default_age_years: float
    default_stance: str


class EloConfig(BaseModel):
    initial: float
    k: float
    finish_multiplier: float
    tuning_start: date
    tuning_end: date
    grid_k: list[float]
    grid_finish_multiplier: list[float]


class LeakageConfig(BaseModel):
    test_years: tuple[int, int]
    max_logloss_gain: float
    max_accuracy_gain: float
    max_identifiability_auc_gain: float
    max_captured_share: float


class ModelsConfig(BaseModel):
    test_years: tuple[int, int]
    inner_val_years: int
    optuna_trials_lgbm: int
    optuna_trials_lr: int
    lgbm_seeds: list[int]
    calibrations: list[str]
    leakage_accuracy_alarm: float
    start_year_experiment: list[int]
    profile_features: list[str]
    reports_dir: Path


class Config(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    http: HttpConfig
    greco: GrecoConfig
    wikipedia: WikipediaConfig
    staleness: StalenessConfig
    names: NamesConfig
    features: FeaturesConfig
    elo: EloConfig
    leakage: LeakageConfig
    models: ModelsConfig


def load_config(path: Path | None = None) -> Config:
    """Load config from ``path``, ``$UFC_CONFIG``, or the default location."""
    path = path or Path(os.environ.get("UFC_CONFIG", DEFAULT_CONFIG_PATH))
    with open(path) as f:
        cfg = Config.model_validate(yaml.safe_load(f))
    cfg.paths = cfg.paths.resolved(PROJECT_ROOT)
    if not cfg.models.reports_dir.is_absolute():
        cfg.models.reports_dir = PROJECT_ROOT / cfg.models.reports_dir
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Process-wide cached config."""
    return load_config()
