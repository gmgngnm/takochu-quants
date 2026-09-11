"""スコア合成とポートフォリオ構築.

重みは config/default.yaml で与える仮説にすぎない。必ず
バックテストで検証すること。触るパラメータは 5 個以内に抑える
（増やすほど過剰最適化で、検証期間だけ綺麗に勝つ曲線が出来る）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINSOR = 3.0


def cross_sectional_z(
    df: pd.DataFrame, column: str, by: list[str], winsor: float = WINSOR
) -> pd.Series:
    """断面内 z-score。外れ値は winsor で切る."""
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    values = pd.to_numeric(df[column], errors="coerce")
    grouped = values.groupby([df[k] for k in by])
    z = (values - grouped.transform("mean")) / grouped.transform("std").replace(0.0, np.nan)
    return z.clip(-winsor, winsor)


def _mean_of(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """利用可能な列だけの平均。全部欠損なら NaN を返す."""
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(np.nan, index=frame.index)
    return frame[present].mean(axis=1, skipna=True)


def build_scores(
    panel: pd.DataFrame,
    weights: dict[str, float],
    sector_field: str = "sector33",
) -> pd.DataFrame:
    """パネルに構成スコアと合成スコアを付与する."""
    df = panel.copy()
    by_date = ["Date"]
    has_sector = sector_field in df.columns and df[sector_field].notna().any()
    by_sector = ["Date", sector_field] if has_sector else by_date

    z = pd.DataFrame(index=df.index)
    z["revision_op"] = cross_sectional_z(df, "revision_op", by_date)
    z["progress_gap"] = cross_sectional_z(df, "progress_gap", by_date)
    z["op_yoy"] = cross_sectional_z(df, "op_yoy", by_date)

    z["roe"] = cross_sectional_z(df, "roe", by_sector)
    z["equity_ratio"] = cross_sectional_z(df, "equity_ratio", by_sector)
    z["op_margin_delta"] = cross_sectional_z(df, "op_margin_delta", by_sector)

    z["mom_12_1"] = cross_sectional_z(df, "mom_12_1", by_date)
    z["ma_gap_60"] = cross_sectional_z(df, "ma_gap_60", by_date)
    z["dist_52w_high"] = cross_sectional_z(df, "dist_52w_high", by_date)

    # 割安ほど高スコアにするため符号を反転する
    z["per_inv"] = -cross_sectional_z(df, "per", by_sector)
    z["pbr_inv"] = -cross_sectional_z(df, "pbr", by_sector)

    df["score_revision"] = _mean_of(z, ["revision_op", "progress_gap", "op_yoy"])
    df["score_quality"] = _mean_of(z, ["roe", "equity_ratio", "op_margin_delta"])
    df["score_momentum"] = _mean_of(z, ["mom_12_1", "ma_gap_60", "dist_52w_high"])
    df["score_value"] = _mean_of(z, ["per_inv", "pbr_inv"])
    df["score_llm"] = (
        cross_sectional_z(df, "llm_score", by_date)
        if "llm_score" in df.columns
        else pd.Series(np.nan, index=df.index)
    )

    components = {
        "revision": "score_revision",
        "quality": "score_quality",
        "momentum": "score_momentum",
        "value": "score_value",
        "llm": "score_llm",
    }

    # 欠損している構成要素の重みは、残りの要素に比例配分する。
    # 0 で埋めると「データが無い銘柄ほど平均的に見える」歪みが出るため。
    total = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for key, column in components.items():
        w = float(weights.get(key, 0.0))
        if w == 0.0:
            continue
        values = df[column]
        mask = values.notna()
        total = total.add((values * w).where(mask, 0.0), fill_value=0.0)
        weight_sum = weight_sum.add(pd.Series(w, index=df.index).where(mask, 0.0), fill_value=0.0)

    df["score"] = (total / weight_sum.replace(0.0, np.nan)).where(weight_sum > 0)
    return df


def select_portfolio(snapshot: pd.DataFrame, config: dict) -> pd.DataFrame:
    """ある判断日のパネルから保有銘柄と目標ウェイトを決める.

    Args:
        snapshot: 単一 Date のパネル（score 済み）。
        config: config["portfolio"]。
    Returns:
        Code, score, weight, sector を持つ DataFrame。
    """
    n = int(config.get("n_positions", 15))
    max_weight = float(config.get("max_weight", 1.0))
    max_sector_weight = float(config.get("max_sector_weight", 1.0))
    sector_field = config.get("sector_field", "Sector33Code")
    sector_col = "sector33" if "sector33" in snapshot.columns else sector_field

    tradable = (
        snapshot["tradable"].astype(bool)
        if "tradable" in snapshot.columns
        else pd.Series(True, index=snapshot.index)
    )
    candidates = snapshot[tradable].copy()
    candidates = candidates.dropna(subset=["score"])
    if candidates.empty:
        return pd.DataFrame(columns=["Code", "score", "weight", "sector"])

    candidates = candidates.sort_values("score", ascending=False)
    sector = (
        candidates[sector_col].astype("string").fillna("UNKNOWN")
        if sector_col in candidates.columns
        else pd.Series("UNKNOWN", index=candidates.index)
    )
    candidates["sector"] = sector

    # 業種上限は銘柄数ベースで近似する（等金額に近い運用なので実質同じ）
    max_per_sector = max(1, int(np.floor(max_sector_weight * n)))
    picked: list[int] = []
    counts: dict[str, int] = {}
    for idx, row in candidates.iterrows():
        if len(picked) >= n:
            break
        key = row["sector"]
        if counts.get(key, 0) >= max_per_sector:
            continue
        picked.append(idx)
        counts[key] = counts.get(key, 0) + 1

    selected = candidates.loc[picked].copy()
    if selected.empty:
        return pd.DataFrame(columns=["Code", "score", "weight", "sector"])

    if config.get("weighting", "equal") == "inverse_vol" and "vol_20" in selected.columns:
        vol = pd.to_numeric(selected["vol_20"], errors="coerce")
        # ボラが取れない銘柄は中央値で代用（除外すると銘柄数が安定しない）
        vol = vol.fillna(vol.median()).replace(0.0, np.nan)
        inv = 1.0 / vol
        weight = inv / inv.sum() if inv.notna().any() else None
    else:
        weight = None

    if weight is None:
        weight = pd.Series(1.0 / len(selected), index=selected.index)

    weight = _cap_weights(weight, max_weight)
    selected["weight"] = weight
    return selected[["Code", "score", "weight", "sector"]].reset_index(drop=True)


def _cap_weights(weight: pd.Series, max_weight: float, iterations: int = 20) -> pd.Series:
    """上限を超えた分を他銘柄に配り直す（合計は 1 のまま）."""
    w = weight.copy()
    if max_weight >= 1.0 or len(w) == 0:
        return w
    # 銘柄数が少なく上限を満たせない場合は等金額に落とす
    if max_weight * len(w) <= 1.0:
        return pd.Series(1.0 / len(w), index=w.index)

    for _ in range(iterations):
        over = w > max_weight + 1e-12
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        room = ~over
        if not room.any():
            break
        w[room] += excess * (w[room] / w[room].sum())
    return w
