"""特徴量のテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.features.fundamental import build_fundamental_facts
from takochu.features.price import add_price_features


def test_会社予想の上方修正が正の値になる():
    rows = []
    for number, (disclosed, forecast) in enumerate(
        [("2023-05-10", "1000"), ("2023-08-10", "1200")], start=1
    ):
        rows.append(
            {
                "DisclosureNumber": str(number),
                "LocalCode": "13010",
                "DisclosedDate": disclosed,
                "DisclosedTime": "12:00",
                "TypeOfCurrentPeriod": "1Q",
                "CurrentPeriodEndDate": disclosed,
                "CurrentFiscalYearEndDate": "2024-03-31",
                "ForecastOperatingProfit": forecast,
                "OperatingProfit": "200",
                "NetSales": "5000",
                "Profit": "140",
                "Equity": "3000",
            }
        )
    facts = build_fundamental_facts(pd.DataFrame(rows))

    assert pd.isna(facts["revision_op"].iloc[0])                 # 初回は比較対象なし
    assert facts["revision_op"].iloc[1] == pytest.approx(0.2)    # 1000 -> 1200


def test_進捗率の乖離は四半期ペース基準で測る():
    # 1Q 終了時点で通期予想の 40% を達成 -> 期待 25% に対して +15%
    rows = [
        {
            "DisclosureNumber": "1",
            "LocalCode": "13010",
            "DisclosedDate": "2023-08-10",
            "DisclosedTime": "12:00",
            "TypeOfCurrentPeriod": "1Q",
            "CurrentPeriodEndDate": "2023-06-30",
            "CurrentFiscalYearEndDate": "2024-03-31",
            "OperatingProfit": "400",
            "ForecastOperatingProfit": "1000",
            "NetSales": "5000",
            "Profit": "280",
            "Equity": "3000",
        }
    ]
    facts = build_fundamental_facts(pd.DataFrame(rows))
    assert facts["progress_gap"].iloc[0] == pytest.approx(0.15)


def test_通期決算の進捗率は情報を持たないので除外する():
    rows = [
        {
            "DisclosureNumber": "1",
            "LocalCode": "13010",
            "DisclosedDate": "2023-05-10",
            "DisclosedTime": "12:00",
            "TypeOfCurrentPeriod": "FY",
            "CurrentPeriodEndDate": "2023-03-31",
            "CurrentFiscalYearEndDate": "2023-03-31",
            "OperatingProfit": "1000",
            "ForecastOperatingProfit": "1000",
            "NetSales": "12000",
            "Profit": "700",
            "Equity": "3000",
        }
    ]
    facts = build_fundamental_facts(pd.DataFrame(rows))
    assert pd.isna(facts["progress_gap"].iloc[0])


def test_前年同期比は同じ四半期どうしで比較する(statements):
    facts = build_fundamental_facts(statements)
    yoy = facts["op_yoy"].dropna()
    assert len(yoy) > 0
    # 合成データは年 5% 成長なので、極端な値にはならないはず
    assert yoy.abs().median() < 1.0


def test_モメンタムは直近1ヶ月を除いて計算する(quotes):
    from takochu.universe import prepare_quotes

    panel = prepare_quotes(quotes)
    out = add_price_features(panel)

    one = out[out["Code"] == out["Code"].iloc[0]].sort_values("Date").reset_index(drop=True)
    i = 400
    expected = one["close"].iloc[i - 21] / one["close"].iloc[i - 245] - 1
    assert one["mom_12_1"].iloc[i] == pytest.approx(expected)


def test_翌営業日の寄り値が執行価格として並ぶ(quotes):
    from takochu.universe import prepare_quotes

    out = add_price_features(prepare_quotes(quotes))
    one = out[out["Code"] == out["Code"].iloc[0]].sort_values("Date").reset_index(drop=True)
    assert one["next_open"].iloc[10] == pytest.approx(one["open"].iloc[11])
    assert np.isnan(one["next_open"].iloc[-1])
