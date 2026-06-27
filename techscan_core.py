# -*- coding: utf-8 -*-
"""
techscan_core.py — 科技股板块扫描器 · 纯计算 / 取数核心（被构建器与 App 共用）。

复用 chip_distribution / holder_concentration 的引擎，新增：
  - 估值取数（百度源 PE(TTM)/总市值/市净率，绕开东财限频）
  - 参数化的筹码评分、综合分、象限判定（供 App 实时调参）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 预算的衰减系数档位（App 用 select_slider 选，瞬时切换）
DECAY_GRID = [0.6, 0.8, 1.0, 1.2, 1.5]
DECAY_LABELS = [str(d) for d in DECAY_GRID]

# 每档要落库的筹码指标
CHIP_METRICS = ["avg_cost", "profit", "conc", "peak", "premium", "dconc"]


# --------------------------------------------------------------------------- #
# 估值取数（百度源）
# --------------------------------------------------------------------------- #
def fetch_valuation(code: str) -> dict:
    """百度源取最新 PE(TTM) / 总市值(亿) / 市净率。失败返回 NaN。"""
    import akshare as ak

    out = {"pe_ttm": np.nan, "mktcap": np.nan, "pb": np.nan}
    mapping = [("市盈率(TTM)", "pe_ttm"), ("总市值", "mktcap"), ("市净率", "pb")]
    for ind, key in mapping:
        try:
            d = ak.stock_zh_valuation_baidu(symbol=code, indicator=ind, period="近一年")
            if d is not None and len(d):
                out[key] = float(d.iloc[-1]["value"])
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# 参数化评分 / 象限（App 实时调参）
# --------------------------------------------------------------------------- #
def chip_clean_score(
    profit: float,
    conc: float,
    dconc: float,
    premium: float,
    conc_lock: float = 0.25,
    profit_lo: float = 40.0,
    profit_hi: float = 88.0,
) -> float:
    """筹码结构干净度 0~100（户数维度之外，单看筹码像不像健康吸筹/控盘）。"""
    s = 0.0
    if dconc == dconc and dconc < 0:            # 集中度在收窄
        s += 15
    if conc == conc:
        if conc < conc_lock:                     # 高度锁仓
            s += 25
        elif conc < conc_lock + 0.10:
            s += 12
    if profit == profit:
        if profit_lo <= profit <= profit_hi:     # 主力浮盈、不深套也不过热
            s += 30
        elif profit > profit_hi:
            s += 8
    if premium == premium:
        if premium >= 0:                         # 现价不低于平均成本
            s += 30
        elif premium >= -8:
            s += 12
    return s


def composite_score(xizou: float, chip: float, w_holder: float = 0.5) -> float:
    """综合分 = w_holder×吸筹分(户数维度) + (1-w_holder)×筹码分(结构维度)。"""
    xz = xizou if xizou == xizou else 0.0
    cp = chip if chip == chip else 0.0
    return round(w_holder * xz + (1.0 - w_holder) * cp, 1)


def quadrant(holder_chg: float, premium: float, h_thr: float = 0.0, p_thr: float = 0.0) -> str:
    """按 户数环比 / 溢价 与可调分界线归四象限。"""
    if holder_chg != holder_chg or premium != premium:
        return "数据缺失"
    down = holder_chg < h_thr          # 户数在降 -> 筹码集中
    up = premium >= p_thr              # 浮盈
    if down and up:
        return "强势控盘"
    if down and not up:
        return "低位吸筹/筑底"
    if (not down) and up:
        return "高位追高派发"
    return "套牢派发"


QUADRANTS = ["强势控盘", "低位吸筹/筑底", "高位追高派发", "套牢派发", "数据缺失"]
QUAD_COLORS = {
    "强势控盘": "#2ca02c",
    "低位吸筹/筑底": "#1f77b4",
    "高位追高派发": "#ff7f0e",
    "套牢派发": "#d62728",
    "数据缺失": "#999999",
}


# --------------------------------------------------------------------------- #
# 由缓存的 base 指标 + 用户参数，实时算出展示用 DataFrame
# --------------------------------------------------------------------------- #
def derive_view(
    metrics: pd.DataFrame,
    decay: str,
    w_holder: float = 0.5,
    conc_lock: float = 0.25,
    profit_lo: float = 40.0,
    profit_hi: float = 88.0,
    h_thr: float = 0.0,
    p_thr: float = 0.0,
) -> pd.DataFrame:
    """根据选定衰减档与评分参数，从缓存的宽表派生展示列。decay 为 DECAY_LABELS 之一。"""
    df = metrics.copy()

    def col(metric):
        name = f"{metric}@{decay}"
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    df["平均成本"] = col("avg_cost").round(2)
    df["获利%"] = (col("profit") * 100).round(1)
    df["集中度"] = col("conc").round(3)
    df["主峰"] = col("peak").round(2)
    df["溢价%"] = (col("premium") * 100).round(1)
    df["集中度环比"] = col("dconc").round(3)

    df["筹码分"] = [
        chip_clean_score(p, c, dc, pr, conc_lock, profit_lo, profit_hi)
        for p, c, dc, pr in zip(df["获利%"], df["集中度"], df["集中度环比"], df["溢价%"])
    ]
    df["综合分"] = [
        composite_score(xz, cp, w_holder) for xz, cp in zip(df["吸筹分"], df["筹码分"])
    ]
    df["象限"] = [
        quadrant(hc, pr, h_thr, p_thr) for hc, pr in zip(df["户数环比%"], df["溢价%"])
    ]
    return df
