#!/usr/bin/env python3
"""合成データでデータレイクを埋める.

J-Quants の契約前でも、パネル構築・バックテスト・スクリーニングの
一連の流れを通しで確認するための足場。ここで出る損益に意味はない
（乱数なので当然）。見るべきは「パイプラインが最後まで通るか」だけ。

    python scripts/seed_synthetic.py --data-dir /tmp/takochu-demo
    TAKOCHU_DATA_DIR=/tmp/takochu-demo takochu features
    TAKOCHU_DATA_DIR=/tmp/takochu-demo takochu backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from takochu.io.store import Store  # noqa: E402
from takochu.universe import SHARES_FIELD  # noqa: E402


def build(
    n_codes: int, start: str, end: str, seed: int, drift: float = 0.0001
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    codes = [f"{1300 + i * 7}0" for i in range(n_codes)]

    quotes, listed, statements = [], [], []
    number = 0

    for i, code in enumerate(codes):
        n = len(days)
        # drift=0 なら完全な無情報データ。IC が 0 付近に落ちることを
        # 確認するためのリーク検出用スイッチ。
        ret = rng.normal((i - n_codes / 2) * drift, 0.018, n)
        close = 1000 * np.exp(np.cumsum(ret))
        volume = rng.integers(300_000, 3_000_000, n).astype(float)
        quotes.append(
            pd.DataFrame(
                {
                    "Date": days.strftime("%Y-%m-%d"),
                    "Code": code,
                    "Open": close * (1 + rng.normal(0, 0.004, n)),
                    "High": close * (1 + abs(rng.normal(0, 0.006, n))),
                    "Low": close * (1 - abs(rng.normal(0, 0.006, n))),
                    "Close": close,
                    "Volume": volume,
                    "TurnoverValue": close * volume,
                    "AdjustmentClose": close,
                    "AdjustmentVolume": volume,
                }
            )
        )

        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            for q, (quarter, month) in enumerate(
                [("1Q", 8), ("2Q", 11), ("3Q", 2), ("FY", 5)], start=1
            ):
                disclosed = pd.Timestamp(f"{year + (1 if month < 6 else 0)}-{month:02d}-10")
                if not (pd.Timestamp(start) <= disclosed <= pd.Timestamp(end)):
                    continue
                number += 1
                sales = (50_000 + i * 3_000) * (1 + rng.normal(0.05, 0.04)) * q / 4
                op = sales * (0.08 + rng.normal(0, 0.015))
                statements.append(
                    {
                        "DisclosureNumber": str(number),
                        "LocalCode": code,
                        "DisclosedDate": disclosed.strftime("%Y-%m-%d"),
                        "DisclosedTime": "15:30" if i % 2 else "13:00",
                        "TypeOfCurrentPeriod": quarter,
                        "CurrentPeriodEndDate": (disclosed - pd.Timedelta(days=40)).strftime(
                            "%Y-%m-%d"
                        ),
                        "CurrentFiscalYearEndDate": f"{year + 1}-03-31",
                        "NetSales": f"{sales:.0f}",
                        "OperatingProfit": f"{op:.0f}",
                        "OrdinaryProfit": f"{op * 1.02:.0f}",
                        "Profit": f"{op * 0.68:.0f}",
                        "EarningsPerShare": f"{op * 0.68 / 1000:.2f}",
                        "TotalAssets": f"{sales * 3:.0f}",
                        "Equity": f"{sales * 1.5:.0f}",
                        "EquityToAssetRatio": "0.5",
                        "BookValuePerShare": f"{sales * 1.5 / 1000:.2f}",
                        "ForecastNetSales": f"{sales * 4:.0f}",
                        "ForecastOperatingProfit": f"{op * 4 * (1 + rng.normal(0, 0.05)):.0f}",
                        "ForecastProfit": f"{op * 4 * 0.68:.0f}",
                        SHARES_FIELD: "1000000000",
                    }
                )

    for day in days[::60]:
        for i, code in enumerate(codes):
            listed.append(
                {
                    "Date": day.strftime("%Y-%m-%d"),
                    "Code": code,
                    "CompanyName": f"テスト{code}",
                    "MarketCode": "0111" if i < int(n_codes * 0.85) else "0112",
                    "Sector17Code": str(1 + i % 6),
                    "Sector33Code": str(3050 + i % 10),
                }
            )

    calendar = pd.DataFrame(
        {"Date": days.strftime("%Y-%m-%d"), "HolidayDivision": "1"}
    )
    return {
        "daily_quotes": pd.concat(quotes, ignore_index=True),
        "statements": pd.DataFrame(statements),
        "listed_info": pd.DataFrame(listed),
        "trading_calendar": calendar,
    }


def _write_disclosure_texts(statements: pd.DataFrame, out_dir: Path, count: int) -> int:
    """ダミーの定性情報テキスト。分析の中身に意味はなく、件数の確認用."""
    out_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "当第{q}四半期におけるわが国経済は、緩やかな回復基調で推移いたしました。"
        "このような状況のもと、当社グループは主力製品の拡販に注力し、"
        "売上高は前年同期を上回りました。原材料価格の上昇は続いておりますが、"
        "生産効率の改善により収益性を維持しております。"
        "通期の見通しにつきましては、現時点で期初予想を据え置いております。"
    )
    for row in statements.head(count).itertuples():
        text = body.format(q=getattr(row, "TypeOfCurrentPeriod", "1Q")[0]) * 3
        (out_dir / f"{row.DisclosureNumber}.txt").write_text(text, encoding="utf-8")
    return min(count, len(statements))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="出力先データディレクトリ")
    parser.add_argument("--codes", type=int, default=120)
    parser.add_argument("--start", default="2019-01-04")
    parser.add_argument("--end", default="2024-12-30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--disclosure-texts",
        type=int,
        default=0,
        help="開示テキストのダミーを何件書き出すか（takochu analyze --dry-run の確認用）",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0001,
        help="銘柄ごとの一定ドリフト。0 にすると予測不能なデータになる",
    )
    args = parser.parse_args()

    store = Store(Path(args.data_dir))
    built = build(args.codes, args.start, args.end, args.seed, args.drift)
    for name, frame in built.items():
        rows = store.upsert(name, frame)
        print(f"{name:<18} {rows:>10,} 行")

    if args.disclosure_texts:
        written = _write_disclosure_texts(
            built["statements"], Path(args.data_dir) / "disclosures", args.disclosure_texts
        )
        print(f"{'disclosures':<18} {written:>10,} 件")

    print(f"\n完了: {args.data_dir}")
    print("これは乱数から作った偽データです。損益に意味はありません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
