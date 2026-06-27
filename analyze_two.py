# -*- coding: utf-8 -*-
"""串联分析 002009 / 300274：户数(吸筹分) + 筹码(主峰/集中度)，并按报告期对齐验证双重锁仓。"""
import numpy as np
import pandas as pd

from chip_distribution import (
    fetch_hist, compute_chip_distribution, chip_metrics, plot_chip_distribution,
)
from holder_concentration import fetch_holder_history, analyze_holder_history

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

SYMBOLS = {"002009": "天奇股份", "300274": "阳光电源"}
N_PERIODS = 6  # 对齐最近 N 个报告期


def analyze_one(code, name):
    print("=" * 78)
    print(f"  {code} {name}")
    print("=" * 78)

    # ---- 工具二：股东户数 / 吸筹分 ----
    hist = analyze_holder_history(fetch_holder_history(code))
    show = hist[["date", "holders", "户数环比%", "涨跌幅%", "标签", "吸筹分"]].tail(N_PERIODS).copy()
    show["date"] = show["date"].dt.date
    print("\n【工具二 · 股东户数与吸筹分（最近{}期）】".format(N_PERIODS))
    print(show.to_string(index=False))

    # ---- 工具一：当前筹码分布 ----
    df = fetch_hist(code, start_date="20220101")
    cur = float(df["close"].iloc[-1])
    g, c = compute_chip_distribution(df, decay_coef=1.0)
    m = chip_metrics(g, c, cur)
    print(f"\n【工具一 · 当前筹码分布】 数据 {len(df)} 根, 末日 {df['date'].iloc[-1].date()}")
    print(f"  现价        : {cur:.2f}")
    print(f"  平均成本    : {m['avg_cost']:.2f}   ({'获利' if cur>=m['avg_cost'] else '亏损'}: 现价{'高于' if cur>=m['avg_cost'] else '低于'}平均成本)")
    print(f"  获利比例    : {m['profit_ratio']*100:.1f}%")
    print(f"  筹码主峰    : {m['peak_price']:.2f}")
    print(f"  90%成本区间 : {m['cost_low']:.2f} ~ {m['cost_high']:.2f}")
    print(f"  集中度      : {m['concentration']:.3f}  (越小越锁仓)")
    plot_chip_distribution(g, c, cur, m, title=f"{code} {name} 筹码分布", savepath=f"chip_{code}.png")
    print(f"  已出图: chip_{code}.png")

    # ---- 串联：按报告期对齐 户数环比 vs 筹码集中度 ----
    rows = []
    prev_conc = None
    for _, r in hist.tail(N_PERIODS).iterrows():
        d = pd.to_datetime(r["date"])
        sub = df[df["date"] <= d]
        if len(sub) < 30:
            continue
        gg, cc = compute_chip_distribution(sub, decay_coef=1.0)
        asof = float(sub["close"].iloc[-1])
        mm = chip_metrics(gg, cc, asof)
        conc = mm["concentration"]
        d_conc = (conc - prev_conc) if prev_conc is not None else np.nan
        rows.append({
            "报告期": d.date(),
            "户数环比%": r["户数环比%"],
            "筹码集中度": round(conc, 3),
            "集中度环比": round(d_conc, 3) if not np.isnan(d_conc) else np.nan,
            "获利比例%": round(mm["profit_ratio"] * 100, 1),
            "主峰价位": round(mm["peak_price"], 2),
        })
        prev_conc = conc
    cross = pd.DataFrame(rows)
    print("\n【串联 · 户数环比 vs 筹码集中度（按报告期对齐）】")
    print(cross.to_string(index=False))

    # 双重锁仓判定：最近一期 户数降 且 集中度环比<0
    last = cross.iloc[-1]
    holder_down = last["户数环比%"] < 0
    conc_down = (not pd.isna(last["集中度环比"])) and last["集中度环比"] < 0
    verdict = "✓ 双重锁仓确认（户数降 + 集中度收窄）" if (holder_down and conc_down) else (
        "户数在降，但集中度未同步收窄" if holder_down else
        "户数未降，不构成吸筹锁仓信号")
    print(f"\n>>> 判定: {verdict}")
    return {"code": code, "name": name, "metrics": m, "cur": cur,
            "holder_last": hist.iloc[-1], "cross": cross}


def main():
    results = []
    for code, name in SYMBOLS.items():
        try:
            results.append(analyze_one(code, name))
        except Exception as e:
            print(f"\n[{code} {name}] 分析失败: {type(e).__name__}: {e}")
        print()
    print("分析完成。图: " + ", ".join(f"chip_{c}.png" for c in SYMBOLS))


if __name__ == "__main__":
    main()
