"""設定ファイルの読み込み."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


@dataclass(frozen=True)
class Config:
    """YAML 設定 + 環境変数由来のパスをまとめたもの."""

    raw: dict[str, Any] = field(default_factory=dict)
    data_dir: Path = REPO_ROOT / "data"

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def holding(self) -> dict[str, Any]:
        return self.raw.get("holding", {})

    @property
    def portfolio(self) -> dict[str, Any]:
        return self.raw["portfolio"]

    @property
    def score_weights(self) -> dict[str, float]:
        return self.raw["score_weights"]

    @property
    def backtest(self) -> dict[str, Any]:
        return self.raw["backtest"]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_dir = os.environ.get("TAKOCHU_DATA_DIR")
    return Config(raw=raw, data_dir=Path(data_dir) if data_dir else REPO_ROOT / "data")
