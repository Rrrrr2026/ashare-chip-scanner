# -*- coding: utf-8 -*-
"""
chip_distribution.py — A 股筹码分布 / 成本分布复刻（逐日筹码转移迭代模型）

不使用回归，使用通达信经典的逐日"筹码转移"迭代模型复刻彩色成本分布柱状图。

核心思想（每个交易日）：
    1. 存量筹码按 g = min(换手率 × 衰减系数, 1) 的比例从【每个价位】等比例清掉
       （= 这部分老筹码"换手卖出"）。
    2. 当天成交量按【三角分布】铺在当日 [low, high] 价格区间，峰值价位取 (H+L+C)/3。
    3. 迭代公式： chips = chips * (1 - g) + tri_normalized * g
    4. 全程结束后把 chips 归一化到 sum = 1。

旋钮 decay_coef（衰减系数）：
    = 1.0  标准；
    < 1.0  老套牢盘更"黏"（历史成本权重高，换手清得慢）；
    > 1.0  更看重近期成本（换手清得快，分布更贴近最近成交价）。

设计约定：取数函数 (fetch_*) 与纯计算函数 (compute_* / chip_metrics) 完全分离，
纯计算函数只接收 DataFrame，方便用合成 / mock 数据做单元测试。
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 中文字体
# --------------------------------------------------------------------------- #
def setup_chinese_font():
    """matplotlib 中文字体设置：优先 Noto Sans CJK SC，回退 SimHei / 文泉驿。"""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        plt.rcParams["font.sans-serif"] = [chosen] + plt.rcParams.get(
            "font.sans-serif", []
        )
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# --------------------------------------------------------------------------- #
# 纯计算：单日三角分布权重
# --------------------------------------------------------------------------- #
def _triangle_weights(grid: np.ndarray, low: float, high: float, close: float) -> np.ndarray:
    """
    把当天成交量按三角分布铺在 [low, high] 上，峰值价位 = (H + L + C) / 3。
    返回与 grid 等长、且 sum == 1 的权重数组。
    """
    grid = np.asarray(grid, dtype=float)
    w = np.zeros_like(grid)

    # 退化：当日没有价格区间（一字板 / 数据异常），全部堆在最接近 close 的格子
    if not np.isfinite(high) or not np.isfinite(low) or high <= low:
        w[np.argmin(np.abs(grid - close))] = 1.0
        return w

    peak = (high + low + close) / 3.0
    peak = min(max(peak, low), high)  # 数值保护

    # 上升边 [low, peak]
    if peak > low:
        mask = (grid >= low) & (grid <= peak)
        w[mask] = (grid[mask] - low) / (peak - low)
    # 下降边 [peak, high]
    if high > peak:
        mask = (grid > peak) & (grid <= high)
        w[mask] = (high - grid[mask]) / (high - peak)
    # 峰值点本身
    if peak <= low or peak >= high:
        w[np.argmin(np.abs(grid - peak))] = max(
            w[np.argmin(np.abs(grid - peak))], 1.0
        )

    w[w < 0] = 0.0
    s = w.sum()
    if s <= 0:
        w[np.argmin(np.abs(grid - close))] = 1.0
        return w
    return w / s


# --------------------------------------------------------------------------- #
# 纯计算：逐日筹码迭代
# --------------------------------------------------------------------------- #
def compute_chip_distribution(
    df: pd.DataFrame,
    decay_coef: float = 1.0,
    n_bins: int = 200,
    price_pad: float = 0.02,
    lock_strength: float = 0.0,
    lock_floor: float = 0.2,
):
    """
    逐日筹码转移迭代，得到当前的筹码（成本）分布。

    参数
    ----
    df : DataFrame，含列 high / low / close / turnover（换手率，小数 0~1），
         按时间【旧 -> 新】排序。
    decay_coef : 衰减系数旋钮。
    n_bins : 价格网格格子数。
    price_pad : 价格网格相对最低 / 最高价向外扩展的比例。
    lock_strength : 「活筹/死筹」锁仓强度 0~1。0=经典均匀换手(老模型)；
         >0 时，深度获利(远低于现价)的筹码被视为庄家/机构锁仓盘、换手清洗更慢，
         当日换手优先从近现价的「活筹」里清掉。直接缓解"低位筹码被误杀"的偏差。
    lock_floor : 锁仓筹码的最低换手保留比例(避免完全不动)，默认 0.2。

    返回
    ----
    (grid, chips) : grid 为价格网格(升序)，chips 为归一化后的筹码占比(sum==1)。
    """
    required = {"high", "low", "close", "turnover"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df 缺少必要列: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("df 为空")

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    turns = df["turnover"].to_numpy(dtype=float)

    pmin = np.nanmin(lows) * (1.0 - price_pad)
    pmax = np.nanmax(highs) * (1.0 + price_pad)
    if pmax <= pmin:
        pmax = pmin + 1e-6
    grid = np.linspace(pmin, pmax, int(n_bins))

    lock_strength = float(np.clip(lock_strength, 0.0, 1.0))
    chips = np.zeros(int(n_bins), dtype=float)
    for h, l, c, t in zip(highs, lows, closes, turns):
        if not np.isfinite(t):
            t = 0.0
        g = min(max(t * decay_coef, 0.0), 1.0)  # 当日换手清洗比例
        tri = _triangle_weights(grid, l, h, c)
        if lock_strength <= 0.0:
            chips = chips * (1.0 - g) + tri * g  # 经典均匀换手
        else:
            # 两速：按价位直接削减——近现价的「活筹」照常 g 清洗，深度获利的「死筹」
            # 按权重 w(lock_floor~1)少清(近乎冻结)。新筹码仍按当日价铺入 g。
            # 跌破当日收盘价约 lock_zone 即视为完全锁仓(死筹)。
            ref = c if c > 0 else (h + l) / 2.0
            lock_zone = 0.5
            depth = np.clip((ref - grid) / (ref * lock_zone), 0.0, 1.0)  # 0=现价及以上, 1=深度获利
            w = np.clip(1.0 - lock_strength * depth, lock_floor, 1.0)     # 活筹权重(死筹→lock_floor)
            chips = chips - chips * (g * w) + tri * g

    s = chips.sum()
    if s > 0:
        chips = chips / s
    return grid, chips


# --------------------------------------------------------------------------- #
# 纯计算：派生指标
# --------------------------------------------------------------------------- #
def chip_metrics(
    grid: np.ndarray,
    chips: np.ndarray,
    current_price: float,
    ratio: float = 0.90,
) -> dict:
    """
    由筹码分布派生关键指标。

    返回 dict：
        avg_cost        平均成本（按筹码加权）
        profit_ratio    获利比例（现价【下方】筹码占比）
        peak_price      筹码主峰价位（占比最高的价格）
        cost_low/high   X% 成本区间上下沿（默认 90%，累计分位法）
        cost_range      区间宽度
        concentration   集中度 = 区间宽 / 区间中值（越小越锁仓）
        ratio           使用的成本区间比例
        current_price   传入的现价
    """
    grid = np.asarray(grid, dtype=float)
    chips = np.asarray(chips, dtype=float)
    total = chips.sum()
    if total <= 0:
        raise ValueError("chips 全为 0，无法计算指标")
    p = chips / total

    avg_cost = float((grid * p).sum())
    profit_ratio = float(p[grid <= current_price].sum())
    peak_price = float(grid[int(np.argmax(p))])

    cum = np.cumsum(p)
    lo_q = (1.0 - ratio) / 2.0
    hi_q = 1.0 - lo_q
    # 累计分位法：在 (cum, grid) 上插值出分位价格
    cost_low = float(np.interp(lo_q, cum, grid))
    cost_high = float(np.interp(hi_q, cum, grid))
    cost_range = cost_high - cost_low
    mid = (cost_high + cost_low) / 2.0
    concentration = float(cost_range / mid) if mid != 0 else float("nan")

    return {
        "current_price": float(current_price),
        "avg_cost": avg_cost,
        "profit_ratio": profit_ratio,
        "peak_price": peak_price,
        "cost_low": cost_low,
        "cost_high": cost_high,
        "cost_range": cost_range,
        "concentration": concentration,
        "ratio": ratio,
    }


# --------------------------------------------------------------------------- #
# 取数（联网）：akshare 日线 -> 标准化 df
#   - 东财源(stock_zh_a_hist) 偶发反爬/限频会主动掐连接(RemoteDisconnected)，
#     故加自动重试；整段失败再回落到新浪源(stock_zh_a_daily)。
#   - 换手率口径不同：东财返回百分数(要 /100)，新浪返回的本就是小数(不用 /100)。
# --------------------------------------------------------------------------- #
_OUT_COLS = ["date", "open", "close", "high", "low", "volume", "turnover"]


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


def _sina_symbol(code: str) -> str:
    """6 位代码 -> 新浪需要的带交易所前缀代码(sh/sz/bj)。已带前缀则原样返回。"""
    s = str(code).strip().lower()
    if s[:2] in ("sh", "sz", "bj"):
        return s
    c = s.zfill(6)
    if c[0] == "6" or c[0] == "9":          # 沪市主板 / 沪 B
        return "sh" + c
    if c[0] in ("0", "3", "2"):             # 深市主板 / 创业板 / 深 B
        return "sz" + c
    if c[0] in ("4", "8"):                  # 北交所
        return "bj" + c
    return "sh" + c


def _normalize_em(raw: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
        "最低": "low", "成交量": "volume", "成交额": "amount", "换手率": "turnover",
    }
    df = raw.rename(columns=colmap)
    df["date"] = pd.to_datetime(df["date"])
    df["turnover"] = df["turnover"].astype(float) / 100.0  # 百分数 -> 小数
    df = df.sort_values("date").reset_index(drop=True)
    return df[_OUT_COLS]


def _normalize_sina(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["turnover"] = df["turnover"].astype(float)  # 新浪本就是小数，不再 /100
    df = df.sort_values("date").reset_index(drop=True)
    return df[_OUT_COLS]


def fetch_hist(
    symbol: str,
    start_date: str = "20230101",
    end_date: str = "20991231",
    adjust: str = "qfq",
    source: str = "auto",
) -> pd.DataFrame:
    """
    取日线并标准化为 compute_chip_distribution 所需的 df（含小数换手率）。

    source : "auto"(默认，东财优先、失败回落新浪) / "em"(仅东财) / "sina"(仅新浪)。
    返回列：date / open / close / high / low / volume / turnover(小数)。
    """
    import akshare as ak

    def _em():
        raw = ak.stock_zh_a_hist(
            symbol=symbol, period="daily",
            start_date=start_date, end_date=end_date, adjust=adjust,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError("东财源返回空")
        return _normalize_em(raw)

    def _sina():
        raw = ak.stock_zh_a_daily(
            symbol=_sina_symbol(symbol),
            start_date=start_date, end_date=end_date, adjust=adjust,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError("新浪源返回空")
        return _normalize_sina(raw)

    if source == "em":
        return _retry(_em)
    if source == "sina":
        return _retry(_sina)

    # auto：东财优先，整段失败再回落新浪
    try:
        return _retry(_em, tries=3)
    except Exception as e_em:  # noqa: BLE001
        try:
            df = _retry(_sina, tries=3)
            print(f"[fetch_hist] 东财源失败({type(e_em).__name__})，已回落新浪源。")
            return df
        except Exception as e_sina:  # noqa: BLE001
            raise RuntimeError(
                f"东财与新浪源均失败 -> em={type(e_em).__name__}: {e_em} | "
                f"sina={type(e_sina).__name__}: {e_sina}"
            )


# --------------------------------------------------------------------------- #
# 画图：横向 barh 彩色成本分布
# --------------------------------------------------------------------------- #
def plot_chip_distribution(
    grid: np.ndarray,
    chips: np.ndarray,
    current_price: float,
    metrics: dict | None = None,
    title: str = "筹码分布",
    savepath: str | None = None,
    show: bool = False,
    ax=None,
):
    """
    横向柱状图：现价【下方】获利盘 / 【上方】套牢盘 两色，标注现价、平均成本、X% 成本区间。
    """
    import matplotlib.pyplot as plt

    setup_chinese_font()
    grid = np.asarray(grid, dtype=float)
    chips = np.asarray(chips, dtype=float)
    if metrics is None:
        metrics = chip_metrics(grid, chips, current_price)

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 10))
        created = True

    bar_h = (grid[1] - grid[0]) * 0.9 if len(grid) > 1 else 0.1
    # 现价下方 = 获利盘（红）；上方 = 套牢盘（绿）。沿用 A 股 红涨绿跌 直觉。
    colors = np.where(grid <= current_price, "#d62728", "#2ca02c")
    ax.barh(grid, chips, height=bar_h, color=colors, alpha=0.85)

    # 90% 成本区间阴影
    ax.axhspan(
        metrics["cost_low"],
        metrics["cost_high"],
        color="#1f77b4",
        alpha=0.08,
        zorder=0,
    )
    xmax = chips.max() if chips.max() > 0 else 1.0

    def _hline(y, color, label):
        ax.axhline(y, color=color, lw=1.4, ls="--")
        ax.text(xmax * 0.98, y, f" {label} {y:.2f}", color=color,
                va="bottom", ha="right", fontsize=9)

    _hline(current_price, "#333333", "现价")
    _hline(metrics["avg_cost"], "#9467bd", "平均成本")
    ax.axhline(metrics["cost_low"], color="#1f77b4", lw=0.9, ls=":")
    ax.axhline(metrics["cost_high"], color="#1f77b4", lw=0.9, ls=":")

    pct = int(round(metrics["ratio"] * 100))
    sub = (
        f"获利比例 {metrics['profit_ratio'] * 100:.1f}% | "
        f"主峰 {metrics['peak_price']:.2f} | "
        f"{pct}% 区间 [{metrics['cost_low']:.2f}, {metrics['cost_high']:.2f}] | "
        f"集中度 {metrics['concentration']:.3f}"
    )
    ax.set_title(f"{title}\n{sub}", fontsize=11)
    ax.set_xlabel("筹码占比")
    ax.set_ylabel("价格")
    ax.margins(y=0.01)

    if savepath:
        plt.tight_layout()
        plt.savefig(savepath, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    elif created:
        plt.close(fig)
    return ax


# --------------------------------------------------------------------------- #
# 用法示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    SYMBOL = "600519"  # 贵州茅台
    try:
        df = fetch_hist(SYMBOL, start_date="20230101")
        print(f"取到 {SYMBOL} 日线 {len(df)} 根，区间 "
              f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
        cur = float(df["close"].iloc[-1])
        grid, chips = compute_chip_distribution(df, decay_coef=1.0)
        m = chip_metrics(grid, chips, cur)
        print("现价        :", round(m["current_price"], 2))
        print("平均成本    :", round(m["avg_cost"], 2))
        print("获利比例    :", f"{m['profit_ratio'] * 100:.1f}%")
        print("筹码主峰    :", round(m["peak_price"], 2))
        print("90% 成本区间:", round(m["cost_low"], 2), "~", round(m["cost_high"], 2))
        print("集中度      :", round(m["concentration"], 3))
        plot_chip_distribution(grid, chips, cur, m,
                               title=f"{SYMBOL} 筹码分布",
                               savepath=f"chip_{SYMBOL}.png")
        print(f"已保存 chip_{SYMBOL}.png")
    except Exception as e:  # 联网 / 接口异常时给出提示，不影响纯计算函数被复用
        print("联网取数失败（不影响纯计算函数）：", repr(e))
