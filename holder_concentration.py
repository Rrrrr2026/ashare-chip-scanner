# -*- coding: utf-8 -*-
"""
holder_concentration.py — 股东户数 + 集中度吸筹追踪器

逻辑：股东户数是带"身份"的公开数据，户数下降 = 筹码向少数人集中。
把【户数环比变化】和【同期股价涨跌幅】交叉打标签，给出 0~100 的吸筹分（50 为中性）。

评分（score，0~100，50 中性）：
    户数分量 holder_comp = clip(-Δ户数% / 0.05 * 30, -30, 40)
        —— 户数每降 5%（Δ = -0.05）给 +30；降得越多越接近上限 +40；上升则扣分。
    价格情景修正 corr：
        户数降>2% 且 涨幅 ≤ 5%        -> "低位吸筹"            +10（最强）
        户数降>2% 且 5% < 涨幅 ≤ 25%  -> "拉升中集中"          0
        户数降>2% 且 涨幅 > 25%       -> "高位集中(已大涨,谨慎)" -25
        户数升>2%                     -> 跌则"派发/散户接盘"、涨则"追高进场"  -20
        其余                          -> "中性"                0
    score = clip(50 + holder_comp + corr, 0, 100)

约定：Δ户数% 与 涨幅 均以【小数】表示（-0.05 = 跌 5%，0.10 = 涨 10%）。
取数函数 (fetch_*) 与纯计算函数 (label_period / holder_score / analyze_holder_history)
分离，纯计算函数只接收 DataFrame，便于用 mock 数据离线测试。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 纯计算：情景打标签
# --------------------------------------------------------------------------- #
def label_period(d_holders: float, price_chg: float):
    """
    根据户数环比变化 d_holders 与同期涨跌幅 price_chg（均为小数）返回 (标签, 价格修正分)。
    """
    if d_holders < -0.02:  # 户数下降超过 2%
        if price_chg <= 0.05:
            return "低位吸筹", 10
        elif price_chg <= 0.25:
            return "拉升中集中", 0
        else:
            return "高位集中(已大涨,谨慎)", -25
    elif d_holders > 0.02:  # 户数上升超过 2%
        if price_chg < 0:
            return "派发/散户接盘", -20
        else:
            return "追高进场", -20
    else:
        return "中性", 0


def holder_score(d_holders: float, price_chg: float):
    """
    返回 (score, label, holder_comp)。d_holders / price_chg 为小数。
    若 d_holders 为 NaN（首期无环比）返回 (nan, "首期", nan)。
    """
    if d_holders is None or (isinstance(d_holders, float) and np.isnan(d_holders)):
        return float("nan"), "首期", float("nan")
    holder_comp = float(np.clip(-d_holders / 0.05 * 30.0, -30.0, 40.0))
    label, corr = label_period(d_holders, price_chg)
    score = float(np.clip(50.0 + holder_comp + corr, 0.0, 100.0))
    return score, label, holder_comp


# --------------------------------------------------------------------------- #
# 纯计算：历史序列分析
# --------------------------------------------------------------------------- #
def analyze_holder_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入一段股东户数历史（按时间【旧 -> 新】排序），列：
        date       截止日
        holders    股东户数
        price_chg  该统计区间的股价涨跌幅（小数）
    计算每期户数环比、标签、吸筹分，返回带分析列的新 DataFrame。
    """
    required = {"date", "holders", "price_chg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df 缺少必要列: {sorted(missing)}")

    out = df.copy().sort_values("date").reset_index(drop=True)
    out["d_holders"] = out["holders"].pct_change()  # 户数环比（小数）

    scores, labels, comps = [], [], []
    for dh, pc in zip(out["d_holders"], out["price_chg"]):
        s, lab, comp = holder_score(dh, pc)
        scores.append(s)
        labels.append(lab)
        comps.append(comp)

    out["户数环比%"] = (out["d_holders"] * 100).round(2)
    out["涨跌幅%"] = (out["price_chg"] * 100).round(2)
    out["标签"] = labels
    out["吸筹分"] = [round(s, 1) if not np.isnan(s) else np.nan for s in scores]
    return out


# --------------------------------------------------------------------------- #
# 取数（联网）：akshare 股东户数明细 -> 标准化 df（列名关键词容错）
# --------------------------------------------------------------------------- #
def _find_col(cols, includes, excludes=()):
    """在列名里按关键词容错匹配：包含全部 includes、且不含任何 excludes。"""
    for c in cols:
        name = str(c)
        if all(k in name for k in includes) and not any(k in name for k in excludes):
            return c
    return None


def _retry(fn, tries: int = 3, delay: float = 1.5, backoff: float = 1.7):
    """对可能瞬时失败的取数调用做带退避的重试。"""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — 取数层统一兜底重试
            last = e
            if i < tries - 1:
                time.sleep(delay)
                delay *= backoff
    raise last


def fetch_holder_history(symbol: str) -> pd.DataFrame:
    """
    用 akshare 取股东户数明细并标准化为 analyze_holder_history 所需的 df。
    akshare 列名偶有变动，用关键词容错匹配【截止日 / 股东户数 / 区间涨跌幅】；
    接口偶发掐连接，故加自动重试。
    """
    import akshare as ak

    raw = _retry(lambda: ak.stock_zh_a_gdhs_detail_em(symbol=symbol))
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"akshare 未返回 {symbol} 的股东户数数据")

    cols = list(raw.columns)
    date_col = (
        _find_col(cols, ["股东户数统计截止日"])
        or _find_col(cols, ["截止日"])
        or _find_col(cols, ["截止"])
        or _find_col(cols, ["日期"])
    )
    holders_col = (
        _find_col(cols, ["股东户数", "本次"])
        or _find_col(cols, ["股东户数"], excludes=["增减", "比例", "上次", "占"])
        or _find_col(cols, ["户数"], excludes=["增减", "比例", "上次"])
    )
    chg_col = _find_col(cols, ["区间涨跌幅"]) or _find_col(cols, ["涨跌幅"])

    if date_col is None or holders_col is None:
        raise RuntimeError(f"无法从列名识别截止日/股东户数，原始列：{cols}")

    df = pd.DataFrame()
    df["date"] = pd.to_datetime(raw[date_col])
    df["holders"] = pd.to_numeric(raw[holders_col], errors="coerce")
    if chg_col is not None:
        df["price_chg"] = pd.to_numeric(raw[chg_col], errors="coerce") / 100.0
    else:
        df["price_chg"] = np.nan

    df = df.dropna(subset=["holders"]).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 批量扫描自选股
# --------------------------------------------------------------------------- #
def scan(symbols, fetch_fn=fetch_holder_history) -> pd.DataFrame:
    """
    批量扫描自选股，取每只票【最新一期】的吸筹分，按分数降序输出。
    fetch_fn 可注入 mock 取数函数用于离线测试。
    """
    rows = []
    for sym in symbols:
        try:
            hist = fetch_fn(sym)
            res = analyze_holder_history(hist)
            last = res.iloc[-1]
            rows.append(
                {
                    "代码": sym,
                    "截止日": pd.to_datetime(last["date"]).date(),
                    "股东户数": int(last["holders"]),
                    "户数环比%": last["户数环比%"],
                    "涨跌幅%": last["涨跌幅%"],
                    "标签": last["标签"],
                    "吸筹分": last["吸筹分"],
                }
            )
        except Exception as e:
            rows.append(
                {
                    "代码": sym,
                    "截止日": None,
                    "股东户数": None,
                    "户数环比%": None,
                    "涨跌幅%": None,
                    "标签": f"取数失败:{type(e).__name__}",
                    "吸筹分": np.nan,
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values("吸筹分", ascending=False, na_position="last").reset_index(
        drop=True
    )


def filter_accumulating(scan_df: pd.DataFrame, min_score: float = 60.0) -> list:
    """
    【两者串联】从 scan 结果里筛出"户数在降 + 吸筹分达标"的票，
    返回代码列表，交给 chip_distribution 进一步看筹码主峰与集中度是否同步收窄。
    """
    mask = (scan_df["户数环比%"] < 0) & (scan_df["吸筹分"] >= min_score)
    return scan_df.loc[mask, "代码"].tolist()


# --------------------------------------------------------------------------- #
# 用法示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    pd.set_option("display.unicode.east_asian_width", True)

    WATCHLIST = ["600519", "000001", "300750"]
    try:
        table = scan(WATCHLIST)
        print("=== 自选股最新一期吸筹分（降序）===")
        print(table.to_string(index=False))

        hot = filter_accumulating(table, min_score=60.0)
        print("\n户数在降且吸筹分达标 ->", hot)
        print("（下一步：对这些代码用 chip_distribution 看主峰价位与集中度是否同步收窄）")
    except Exception as e:
        print("联网取数失败（不影响纯计算函数）：", repr(e))
