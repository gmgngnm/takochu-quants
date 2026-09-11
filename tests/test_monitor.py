"""保有中の手仕舞い監視のテスト."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from takochu.monitor import check_holdings


def _panel(paths: dict[str, list[float]], start="2024-06-03") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(next(iter(paths.values()))))
    rows = []
    for code, closes in paths.items():
        close = np.array(closes, dtype="float64")
        rows.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Code": code,
                    "open": close,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "atr_pct": 0.02,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _holdings(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Code": codes, "weight": [1 / len(codes)] * len(codes)})


def test_損切り水準に達した銘柄を拾う():
    # 判断日の翌営業日にエントリー(100)し、その後 -6% まで下げる
    panel = _panel({"A": [99, 100, 98, 96, 94]})
    alerts, status = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-03"), {"stop_loss": 0.05}
    )
    assert len(alerts) == 1
    assert alerts[0].reason == "stop_loss"
    assert alerts[0].entry_price == pytest.approx(100.0)
    assert status.loc[0, "change"] == pytest.approx(-0.06)


def test_利確水準に達した銘柄を拾う():
    panel = _panel({"A": [99, 100, 104, 109, 108]})
    alerts, _ = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-03"), {"take_profit": 0.08}
    )
    assert len(alerts) == 1
    assert alerts[0].reason == "take_profit"


def test_水準に届かなければアラートは出ない():
    panel = _panel({"A": [99, 100, 101, 102, 101]})
    alerts, status = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-03"),
        {"stop_loss": 0.05, "take_profit": 0.08},
    )
    assert alerts == []
    assert len(status) == 1


def test_終値が戻っていてもザラ場で触れていれば拾う():
    """逆指値を置いていれば約定しているはず。終値だけ見ると見落とす."""
    panel = _panel({"A": [99, 100, 100, 100, 100]})
    # 3日目だけ安値が大きく下がったことにする
    mask = (panel["Code"] == "A") & (panel["Date"] == pd.Timestamp("2024-06-05"))
    panel.loc[mask, "low"] = 93.0

    alerts, _ = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-03"), {"stop_loss": 0.05}
    )
    assert len(alerts) == 1
    assert alerts[0].touched_on == "2024-06-05"


def test_損切りが先に並ぶ():
    panel = _panel({"A": [99, 100, 110, 112, 111], "B": [99, 100, 96, 93, 92]})
    alerts, _ = check_holdings(
        panel, _holdings(["A", "B"]), pd.Timestamp("2024-06-03"),
        {"stop_loss": 0.05, "take_profit": 0.08},
    )
    assert [a.code for a in alerts] == ["B", "A"]


def test_ATR基準と固定率は浅いほうを採る():
    """意図した最大損失を超えないこと。ATR 2倍=4% と固定5% なら 4% 側."""
    panel = _panel({"A": [99, 100, 100, 95.8, 95.8]})
    alerts, _ = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-03"),
        {"stop_loss": 0.05, "stop_loss_atr": 2.0},
    )
    assert len(alerts) == 1   # -4.2% は 4% を割るが 5% には届かない


def test_翌営業日が来ていなければ何も返さない():
    panel = _panel({"A": [100, 101]})
    alerts, status = check_holdings(
        panel, _holdings(["A"]), pd.Timestamp("2024-06-04"), {"stop_loss": 0.05}
    )
    assert alerts == [] and status.empty
