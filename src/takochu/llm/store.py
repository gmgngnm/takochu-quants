"""分析結果の永続化と、特徴量パネルへの接続.

LLM 呼び出しは金がかかるので、開示 1 件 × プロンプト版 × モデル で
結果をディスクに残し、再実行では呼ばない。

facts テーブルへの変換では、決算短信と同じ `available_date` を使う。
**LLM スコアも他のファンダ特徴量とまったく同じ PIT 規約に従う** ので、
ここから未来の情報が漏れることはない。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from takochu.llm.prompts import PROMPT_VERSION
from takochu.llm.schema import DisclosureAnalysis
from takochu.pit import add_available_date

log = logging.getLogger(__name__)


class AnalysisCache:
    """JSONL 追記型の結果キャッシュ."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # 書き込み途中で落ちた行は捨てる（追記型なので末尾のみ）
                    log.warning("%s に壊れた行があります。無視します。", self.path)
                    continue
                self._entries[row["cache_key"]] = row
        log.info("キャッシュを読み込みました: %d 件", len(self._entries))

    def __contains__(self, cache_key: str) -> bool:
        return cache_key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, cache_key: str) -> dict | None:
        return self._entries.get(cache_key)

    def put(
        self,
        cache_key: str,
        disclosure_number: str,
        code: str,
        model: str,
        analysis: DisclosureAnalysis,
    ) -> None:
        row = {
            "cache_key": cache_key,
            "disclosure_number": str(disclosure_number),
            "code": str(code),
            "model": model,
            "prompt_version": PROMPT_VERSION,
            **analysis.to_dict(),
            "llm_score": analysis.score,
        }
        self._entries[cache_key] = row
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def to_frame(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame()
        return pd.DataFrame(list(self._entries.values()))


def build_llm_facts(
    analyses: pd.DataFrame,
    statements: pd.DataFrame,
    trading_days: pd.Series | None = None,
) -> pd.DataFrame:
    """分析結果を、開示日ベースの PIT facts テーブルにする.

    Returns:
        LocalCode, available_date, llm_score, llm_confidence,
        llm_quality, llm_tone, llm_risk_count を持つ DataFrame。
    """
    if analyses is None or analyses.empty or statements.empty:
        return pd.DataFrame()

    disclosures = add_available_date(statements, trading_days)[
        ["DisclosureNumber", "LocalCode", "available_date"]
    ].copy()
    disclosures["DisclosureNumber"] = disclosures["DisclosureNumber"].astype(str)
    disclosures["LocalCode"] = disclosures["LocalCode"].astype(str)

    df = analyses.copy()
    df["disclosure_number"] = df["disclosure_number"].astype(str)

    # 同じ開示を両モデルで分析した場合、精査（Opus）の結果を優先する。
    if "model" in df.columns:
        priority = df["model"].str.contains("opus", case=False, na=False).astype(int)
        df = (
            df.assign(_priority=priority)
            .sort_values(["disclosure_number", "_priority"])
            .drop_duplicates("disclosure_number", keep="last")
            .drop(columns="_priority")
        )

    merged = df.merge(
        disclosures,
        left_on="disclosure_number",
        right_on="DisclosureNumber",
        how="inner",
    )
    if merged.empty:
        log.warning("分析結果と決算短信が 1 件も突合しませんでした")
        return pd.DataFrame()

    merged["llm_risk_count"] = merged["risk_flags"].apply(
        lambda flags: len(flags) if isinstance(flags, (list, tuple)) else 0
    )
    out = merged.rename(
        columns={
            "confidence": "llm_confidence",
            "earnings_quality": "llm_quality",
            "guidance_tone": "llm_tone",
        }
    )
    columns = [
        "LocalCode", "available_date", "llm_score", "llm_confidence",
        "llm_quality", "llm_tone", "llm_risk_count",
    ]
    return out[columns].sort_values(["LocalCode", "available_date"]).reset_index(drop=True)


def select_for_detail(
    panel: pd.DataFrame, top_n: int, as_of: pd.Timestamp | None = None
) -> set[str]:
    """精査（Opus）に回す銘柄を、1 次スコアの上位から選ぶ.

    全銘柄を上位モデルに流すのは無駄なので、定量スコアで絞ってから精査する。
    """
    if panel.empty or "score" not in panel.columns:
        return set()
    as_of = as_of or panel["Date"].max()
    snapshot = panel[panel["Date"] == as_of]
    if "tradable" in snapshot.columns:
        snapshot = snapshot[snapshot["tradable"].astype(bool)]
    top = snapshot.dropna(subset=["score"]).nlargest(top_n, "score")
    return set(top["Code"].astype(str))
