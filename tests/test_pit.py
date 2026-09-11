"""PIT 処理のテスト。ここが壊れるとバックテストの結果が全部嘘になる."""

from __future__ import annotations

import pandas as pd
import pytest

from takochu.pit import add_available_date, asof_latest, assert_no_lookahead


def _statement(disclosed: str, time: str | None, code: str = "13010") -> dict:
    return {
        "DisclosureNumber": "1",
        "LocalCode": code,
        "DisclosedDate": disclosed,
        "DisclosedTime": time,
        "OperatingProfit": "100",
    }


def test_引け前の開示は当日から使える():
    df = pd.DataFrame([_statement("2023-05-10", "12:00")])
    out = add_available_date(df)
    assert out["available_date"].iloc[0] == pd.Timestamp("2023-05-10")


def test_引け後の開示は翌営業日から使える():
    days = pd.bdate_range("2023-05-08", "2023-05-20")
    # 金曜 15:30 の開示 -> 翌月曜が最初に使える営業日
    df = pd.DataFrame([_statement("2023-05-12", "15:30")])
    out = add_available_date(df, days)
    assert out["available_date"].iloc[0] == pd.Timestamp("2023-05-15")


def test_開示時刻が欠損なら保守側に倒す():
    """時刻不明を引け前扱いにすると look-ahead になるので引け後に倒す."""
    days = pd.bdate_range("2023-05-08", "2023-05-20")
    df = pd.DataFrame([_statement("2023-05-10", None)])
    out = add_available_date(df, days)
    assert out["available_date"].iloc[0] == pd.Timestamp("2023-05-11")


def test_asof_join_は未来の開示を貼らない():
    panel = pd.DataFrame(
        {
            "Code": ["13010"] * 4,
            "Date": pd.to_datetime(["2023-05-09", "2023-05-10", "2023-05-11", "2023-05-12"]),
        }
    )
    facts = pd.DataFrame(
        {
            "LocalCode": ["13010", "13010"],
            "available_date": pd.to_datetime(["2023-05-10", "2023-05-12"]),
            "op_yoy": [0.10, 0.25],
        }
    )
    merged = asof_latest(panel, facts, ["op_yoy", "available_date"]).sort_values("Date")

    assert pd.isna(merged["op_yoy"].iloc[0])          # 開示前は空
    assert merged["op_yoy"].iloc[1] == pytest.approx(0.10)
    assert merged["op_yoy"].iloc[2] == pytest.approx(0.10)  # 次の開示までは据え置き
    assert merged["op_yoy"].iloc[3] == pytest.approx(0.25)
    assert_no_lookahead(merged)


def test_未来の情報が混ざっていれば検出する():
    bad = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2023-05-10"]),
            "available_date": pd.to_datetime(["2023-05-11"]),
        }
    )
    with pytest.raises(AssertionError, match="未来の情報"):
        assert_no_lookahead(bad)
