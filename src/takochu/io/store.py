"""Parquet データレイクと DuckDB による問い合わせ.

設計方針:
  - 生データは J-Quants から返ってきた形のまま保存する（加工は後段で行う）。
    仕様変更や特徴量のバグがあっても再取得せずに作り直せるようにするため。
  - 各データセットは主キーを持ち、upsert で冪等に追記できる。
    取り込みを何度流しても結果が変わらないことが再現性の前提になる。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    primary_key: tuple[str, ...]
    partition_by_year: str | None = None  # 年でパーティションする日付カラム


DATASETS: dict[str, DatasetSpec] = {
    # 銘柄マスタ。J-Quants は基準日付きで返すので履歴として貯める
    # （過去のユニバースを当時の市場区分で再現するのに必要）。
    "listed_info": DatasetSpec("listed_info", ("Date", "Code")),
    "daily_quotes": DatasetSpec("daily_quotes", ("Date", "Code"), partition_by_year="Date"),
    # 同じ決算に対する訂正開示があるため DisclosureNumber まで含めて一意にする。
    "statements": DatasetSpec("statements", ("DisclosureNumber",)),
    "weekly_margin": DatasetSpec("weekly_margin", ("Date", "Code")),
    # 決算発表予定日。実運用でのブラックアウト判定に必須
    # （過去データは statements の開示日から復元する）。
    "announcement": DatasetSpec("announcement", ("Date", "Code")),
    "topix": DatasetSpec("topix", ("Date",)),
    "trading_calendar": DatasetSpec("trading_calendar", ("Date",)),
}


class Store:
    """data/raw 以下の Parquet 群を読み書きする."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.derived_dir = self.data_dir / "derived"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 配置

    def _paths(self, name: str) -> list[Path]:
        spec = DATASETS[name]
        if spec.partition_by_year:
            return sorted((self.raw_dir / name).glob("*.parquet"))
        path = self.raw_dir / f"{name}.parquet"
        return [path] if path.exists() else []

    # ------------------------------------------------------------------ 書き込み

    def upsert(self, name: str, df: pd.DataFrame) -> int:
        """主キーで重複排除しつつ追記する。戻り値は保存後の総行数."""
        if df is None or df.empty:
            log.info("%s: 書き込む行がありません", name)
            return self.row_count(name)

        spec = DATASETS[name]
        missing = [k for k in spec.primary_key if k not in df.columns]
        if missing:
            raise ValueError(f"{name}: 主キー列が欠けています: {missing}")

        if spec.partition_by_year:
            total = 0
            years = pd.to_datetime(df[spec.partition_by_year]).dt.year
            for year, chunk in df.groupby(years):
                total += self._upsert_file(
                    self.raw_dir / name / f"{year}.parquet", chunk, spec.primary_key
                )
            return total

        return self._upsert_file(self.raw_dir / f"{name}.parquet", df, spec.primary_key)

    @staticmethod
    def _upsert_file(path: Path, df: pd.DataFrame, primary_key: tuple[str, ...]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            # 後から来たデータを正とする（訂正開示・確定値で上書きされる）
            merged = pd.concat([existing, df], ignore_index=True)
        else:
            merged = df.copy()

        merged = merged.drop_duplicates(subset=list(primary_key), keep="last")
        merged = merged.sort_values(list(primary_key)).reset_index(drop=True)
        merged.to_parquet(path, index=False)
        return len(merged)

    def write_derived(self, name: str, df: pd.DataFrame) -> Path:
        """特徴量パネルなど、生成物を保存する（毎回作り直すので上書き）."""
        path = self.derived_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        return path

    def read_derived(self, name: str) -> pd.DataFrame:
        path = self.derived_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} がありません。先に `takochu features` を実行してください。"
            )
        return pd.read_parquet(path)

    # ------------------------------------------------------------------ 読み出し

    def read(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        paths = self._paths(name)
        if not paths:
            return pd.DataFrame()
        frames = [pd.read_parquet(p, columns=columns) for p in paths]
        return pd.concat(frames, ignore_index=True)

    def row_count(self, name: str) -> int:
        paths = self._paths(name)
        if not paths:
            return 0
        con = duckdb.connect()
        try:
            files = [str(p) for p in paths]
            return con.execute(
                "SELECT COUNT(*) FROM read_parquet($files)", {"files": files}
            ).fetchone()[0]
        finally:
            con.close()

    def max_date(self, name: str, column: str = "Date") -> str | None:
        """取り込み済みの最終日。差分取得の開始点を決めるのに使う."""
        paths = self._paths(name)
        if not paths:
            return None
        con = duckdb.connect()
        try:
            files = [str(p) for p in paths]
            value = con.execute(
                f'SELECT MAX("{column}") FROM read_parquet($files)', {"files": files}
            ).fetchone()[0]
            return str(value)[:10] if value is not None else None
        finally:
            con.close()

    def query(self, sql: str, **frames: pd.DataFrame) -> pd.DataFrame:
        """DuckDB で SQL を実行する。ASOF JOIN など PIT 結合に使う."""
        con = duckdb.connect()
        try:
            for key, frame in frames.items():
                con.register(key, frame)
            return con.execute(sql).fetchdf()
        finally:
            con.close()

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in DATASETS:
            date_column = "DisclosedDate" if name == "statements" else "Date"
            rows.append(
                {
                    "dataset": name,
                    "rows": self.row_count(name),
                    "max_date": self.max_date(name, date_column),
                }
            )
        return pd.DataFrame(rows)
