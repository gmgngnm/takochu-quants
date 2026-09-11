"""分析対象テキストの供給.

**重要な制約**: J-Quants の `/fins/statements` が返すのは XBRL 由来の
数値項目で、決算短信の「定性的情報」本文は含まれていない。
つまり LLM 層に食わせるテキストは別経路で用意する必要がある。

当面は、外部で取得したテキストをファイルとして置く方式にしている:

    data/disclosures/<DisclosureNumber>.txt

取得経路は EDINET API（有価証券報告書・四半期報告書の本文）か、
TDnet の適時開示 PDF からの抽出になる。ここを自動化するのはフェーズ3。
テキストが無い開示は単に分析対象から外れる（欠損として扱われる）ので、
一部だけ揃っている状態でも動く。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from takochu.llm.analyzer import DisclosureInput

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200  # これ未満は定型ヘッダだけの可能性が高い


def iter_disclosure_inputs(
    statements: pd.DataFrame,
    text_dir: Path,
    codes: set[str] | None = None,
    since: str | None = None,
) -> Iterator[DisclosureInput]:
    """テキストが用意されている開示だけを DisclosureInput にして返す."""
    text_dir = Path(text_dir)
    if not text_dir.exists():
        log.warning("%s がありません。分析対象は 0 件です。", text_dir)
        return

    df = statements.copy()
    df["LocalCode"] = df["LocalCode"].astype(str)
    df["DisclosureNumber"] = df["DisclosureNumber"].astype(str)
    if since:
        df = df[pd.to_datetime(df["DisclosedDate"]) >= pd.Timestamp(since)]
    if codes:
        df = df[df["LocalCode"].isin(codes)]

    missing = 0
    for row in df.itertuples():
        path = text_dir / f"{row.DisclosureNumber}.txt"
        if not path.exists():
            missing += 1
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) < MIN_TEXT_CHARS:
            missing += 1
            continue
        yield DisclosureInput(
            disclosure_number=row.DisclosureNumber,
            code=row.LocalCode,
            period=str(getattr(row, "CurrentPeriodEndDate", "") or ""),
            document_type=str(getattr(row, "TypeOfCurrentPeriod", "") or ""),
            text=text,
        )

    if missing:
        log.info("テキスト未整備のため %d 件をスキップしました", missing)
