# -*- coding: utf-8 -*-
"""
techscan_build.py — 科技股板块扫描器 · 数据构建器（慢、联网、断点续跑 + 增量检查点）。

把全板块的日线/换手、股东户数、估值抓下来，预算 5 档衰减系数的筹码指标，
缓存成本地 parquet 供 App 离线秒开。

数据源（均绕开东财行情限频）：
  日线/换手 -> 新浪源 ; 股东户数+总市值 -> 东财数据中心(一次调用) ; PE(TTM)/PB/当前总市值 -> 百度源
名单来源：techscan_universe.csv（申万一级 电子+计算机+通信 官方成分，python 生成）。

用法：
    python techscan_build.py                 # 断点续跑（已抓过的跳过），每 25 只落盘一次
    python techscan_build.py --refresh       # 全量重抓
    python techscan_build.py --limit 50      # 只跑前 50 只（试跑）
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import akshare as ak

from chip_distribution import fetch_hist, compute_chip_distribution, chip_metrics
from holder_concentration import analyze_holder_history, _find_col, _retry
from techscan_core import DECAY_GRID, DECAY_LABELS, fetch_valuation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "techscan_data")
METRICS_PARQUET = os.path.join(DATA_DIR, "metrics.parquet")
PRICES_PARQUET = os.path.join(DATA_DIR, "prices.parquet")
UNIVERSE_CSV = os.path.join(HERE, "techscan_universe.csv")
CHECKPOINT = 25  # 每多少只落盘一次

DEFAULT_UNIVERSE = [
    ("000063", "中兴通讯"), ("002415", "海康威视"), ("002230", "科大讯飞"),
    ("002475", "立讯精密"), ("300059", "东方财富"), ("000725", "京东方A"),
]


def load_universe(path: str = UNIVERSE_CSV) -> pd.DataFrame:
    if os.path.exists(path):
        u = pd.read_csv(path, dtype={"code": str})
        u["code"] = u["code"].str.zfill(6)
        if "name" not in u.columns:
            u["name"] = u["code"]
        return u
    u = pd.DataFrame(DEFAULT_UNIVERSE, columns=["code", "name"])
    u.to_csv(path, index=False, encoding="utf-8-sig")
    return u


def _conc_at(df, date, decay):
    sub = df[df["date"] <= pd.to_datetime(date)]
    if len(sub) < 30:
        return np.nan
    g, c = compute_chip_distribution(sub, decay_coef=decay)
    return chip_metrics(g, c, float(sub["close"].iloc[-1]))["concentration"]


def _holder_and_mktcap(code):
    """一次调用东财股东户数接口，返回 (标准化户数df, 报告期总市值[亿])。"""
    raw = _retry(lambda: ak.stock_zh_a_gdhs_detail_em(symbol=code))
    if raw is None or len(raw) == 0:
        raise RuntimeError("股东户数返回空")
    cols = list(raw.columns)
    dcol = (_find_col(cols, ["股东户数统计截止日"]) or _find_col(cols, ["截止日"])
            or _find_col(cols, ["截止"]) or _find_col(cols, ["日期"]))
    hcol = (_find_col(cols, ["股东户数", "本次"])
            or _find_col(cols, ["股东户数"], excludes=["增减", "比例", "上次", "占"]))
    ccol = _find_col(cols, ["区间涨跌幅"]) or _find_col(cols, ["涨跌幅"])
    mvcol = _find_col(cols, ["总市值"])
    if dcol is None or hcol is None:
        raise RuntimeError(f"无法识别户数列：{cols}")

    std = pd.DataFrame({
        "date": pd.to_datetime(raw[dcol]),
        "holders": pd.to_numeric(raw[hcol], errors="coerce"),
        "price_chg": (pd.to_numeric(raw[ccol], errors="coerce") / 100.0) if ccol else np.nan,
    }).dropna(subset=["holders"]).sort_values("date").reset_index(drop=True)

    mktcap = np.nan
    if mvcol is not None:
        s = pd.to_numeric(raw[mvcol], errors="coerce").dropna()
        if len(s):
            mktcap = float(s.iloc[-1]) / 1e8  # 元 -> 亿
    return std, mktcap


def build_one(code: str, name: str, start_date: str = "20220101"):
    row = {"代码": code, "名称": name}

    # 户数 + 报告期总市值（一次调用）
    std, mktcap_report = _holder_and_mktcap(code)
    h = analyze_holder_history(std)
    last = h.iloc[-1]
    row["截止日"] = pd.to_datetime(last["date"]).date()
    row["股东户数"] = int(last["holders"]) if last["holders"] == last["holders"] else None
    row["户数环比%"] = last["户数环比%"]
    row["吸筹分"] = last["吸筹分"]
    row["标签"] = last["标签"]
    holder_dates = list(h["date"].tail(2))

    # 价格 / 筹码（5 档衰减预算）
    df = fetch_hist(code, start_date=start_date, source="sina")
    cur = float(df["close"].iloc[-1])
    row["现价"] = round(cur, 2)
    row["末日"] = df["date"].iloc[-1].date()
    for d, lab in zip(DECAY_GRID, DECAY_LABELS):
        g, c = compute_chip_distribution(df, decay_coef=d)
        m = chip_metrics(g, c, cur)
        row[f"avg_cost@{lab}"] = m["avg_cost"]
        row[f"profit@{lab}"] = m["profit_ratio"]
        row[f"conc@{lab}"] = m["concentration"]
        row[f"peak@{lab}"] = m["peak_price"]
        row[f"premium@{lab}"] = cur / m["avg_cost"] - 1.0
        dconc = np.nan
        if len(holder_dates) == 2:
            cp, cn = _conc_at(df, holder_dates[0], d), _conc_at(df, holder_dates[1], d)
            if cp == cp and cn == cn:
                dconc = cn - cp
        row[f"dconc@{lab}"] = dconc

    # 估值（百度当前值；总市值缺失回落报告期 gdhs 值）
    val = fetch_valuation(code)
    row["pe_ttm"] = val["pe_ttm"]
    row["pb"] = val["pb"]
    row["mktcap"] = val["mktcap"] if val["mktcap"] == val["mktcap"] else mktcap_report

    prices = df[["date", "close", "high", "low", "turnover"]].copy()
    prices.insert(0, "代码", code)
    return row, prices


def _flush(buf_rows, buf_prices):
    """把缓冲区合并进磁盘上的 parquet（增量检查点，安全可续跑）。"""
    if not buf_rows:
        return
    m_old = pd.read_parquet(METRICS_PARQUET) if os.path.exists(METRICS_PARQUET) else pd.DataFrame()
    m = pd.concat([m_old, pd.DataFrame(buf_rows)], ignore_index=True).drop_duplicates("代码", keep="last")
    m.to_parquet(METRICS_PARQUET, index=False)
    if buf_prices:
        p_old = pd.read_parquet(PRICES_PARQUET) if os.path.exists(PRICES_PARQUET) else pd.DataFrame()
        p = pd.concat([p_old] + buf_prices, ignore_index=True).drop_duplicates(["代码", "date"], keep="last")
        p.to_parquet(PRICES_PARQUET, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--universe", default=UNIVERSE_CSV)
    ap.add_argument("--start", default="20220101")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    uni = load_universe(args.universe)
    if args.limit:
        uni = uni.head(args.limit)
    print(f"板块范围共 {len(uni)} 只")

    done = set()
    if not args.refresh and os.path.exists(METRICS_PARQUET):
        done = set(pd.read_parquet(METRICS_PARQUET, columns=["代码"])["代码"].astype(str))
        print(f"已缓存 {len(done)} 只，断点续跑")
    if args.refresh and os.path.exists(METRICS_PARQUET):
        os.remove(METRICS_PARQUET)
        if os.path.exists(PRICES_PARQUET):
            os.remove(PRICES_PARQUET)

    todo = uni if args.refresh else uni[~uni["code"].isin(done)]
    todo = todo.reset_index(drop=True)
    print(f"待抓 {len(todo)} 只\n")

    buf_rows, buf_prices, fails, ok = [], [], [], 0
    for i, r in todo.iterrows():
        code, name = r["code"], r["name"]
        try:
            row, prices = build_one(code, name, start_date=args.start)
            buf_rows.append(row)
            buf_prices.append(prices)
            ok += 1
            print(f"  [{i+1}/{len(todo)}] OK  {code} {name}")
        except Exception as e:
            fails.append((code, name, type(e).__name__))
            print(f"  [{i+1}/{len(todo)}] FAIL {code} {name} -> {type(e).__name__}: {str(e)[:70]}")
        if (i + 1) % CHECKPOINT == 0:
            _flush(buf_rows, buf_prices)
            buf_rows, buf_prices = [], []
            print(f"  -- 检查点已落盘（累计成功 {ok}，失败 {len(fails)}）--")
    _flush(buf_rows, buf_prices)

    total = len(pd.read_parquet(METRICS_PARQUET, columns=["代码"])) if os.path.exists(METRICS_PARQUET) else 0
    print(f"\n完成。本轮成功 {ok}，失败 {len(fails)}；库内合计 {total} 只 -> {METRICS_PARQUET}")
    if fails:
        print("失败清单: " + ", ".join(f"{c}{n}({e})" for c, n, e in fails[:40]))
    print("下一步：streamlit run techscan_app.py")


if __name__ == "__main__":
    main()
