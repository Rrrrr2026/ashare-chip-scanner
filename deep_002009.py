# -*- coding: utf-8 -*-
"""002009 天奇股份 深度拆解：户数/户均持股趋势 + 多衰减筹码 + 集中度演变 + 关键价位。"""
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt

from chip_distribution import (
    fetch_hist, compute_chip_distribution, chip_metrics,
    plot_chip_distribution, setup_chinese_font, _retry,
)
from holder_concentration import fetch_holder_history, analyze_holder_history

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 240)

CODE, NAME = "002009", "天奇股份"


def main():
    # ---------- 1. 股东户数 + 户均持股 全历史趋势 ----------
    raw = _retry(lambda: ak.stock_zh_a_gdhs_detail_em(symbol=CODE))
    raw = raw.rename(columns={
        "股东户数统计截止日": "date", "股东户数-本次": "holders",
        "户均持股数量": "户均持股", "区间涨跌幅": "price_chg",
    })
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").reset_index(drop=True)

    h = analyze_holder_history(
        raw[["date", "holders", "price_chg"]].assign(price_chg=raw["price_chg"] / 100.0)
    )
    h["户均持股"] = raw["户均持股"].values
    print("=" * 86)
    print(f"  {CODE} {NAME} · 股东户数 / 户均持股 趋势（最近10期）")
    print("=" * 86)
    show = h[["date", "holders", "户均持股", "户数环比%", "涨跌幅%", "标签", "吸筹分"]].tail(10).copy()
    show["date"] = show["date"].dt.date
    show["户均持股"] = show["户均持股"].round(0).astype(int)
    print(show.to_string(index=False))

    # ---------- 2. 近期价格走势 ----------
    df = fetch_hist(CODE, start_date="20220101")
    cur = float(df["close"].iloc[-1])
    c20 = float(df["close"].iloc[-21]); c60 = float(df["close"].iloc[-61])
    hi52 = float(df["close"].tail(250).max()); lo52 = float(df["close"].tail(250).min())
    print("\n" + "=" * 86)
    print("  近期价格走势")
    print("=" * 86)
    print(f"  现价 {cur:.2f} | 20日 {(cur/c20-1)*100:+.1f}% | 60日 {(cur/c60-1)*100:+.1f}% | "
          f"近一年区间 {lo52:.2f}~{hi52:.2f} | 当前处区间 {(cur-lo52)/(hi52-lo52)*100:.0f}% 分位")

    # ---------- 3. 三档衰减系数的筹码稳健性 ----------
    print("\n" + "=" * 86)
    print("  三档衰减系数下的筹码指标（稳健性）")
    print("=" * 86)
    rows = []
    chips_by_decay = {}
    for dc in (0.6, 1.0, 1.5):
        g, c = compute_chip_distribution(df, decay_coef=dc)
        m = chip_metrics(g, c, cur)
        chips_by_decay[dc] = (g, c, m)
        rows.append({
            "衰减": dc, "平均成本": round(m["avg_cost"], 2), "主峰": round(m["peak_price"], 2),
            "获利比例%": round(m["profit_ratio"] * 100, 1), "集中度": round(m["concentration"], 3),
            "90%下沿": round(m["cost_low"], 2), "90%上沿": round(m["cost_high"], 2),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 4. 集中度 / 获利比例 逐期演变（对齐户数报告期） ----------
    print("\n" + "=" * 86)
    print("  筹码集中度 / 获利比例 逐期演变（decay=1.0, 对齐报告期）")
    print("=" * 86)
    rows = []; prev = None
    for _, r in h.tail(8).iterrows():
        d = pd.to_datetime(r["date"]); sub = df[df["date"] <= d]
        if len(sub) < 30:
            continue
        g, c = compute_chip_distribution(sub, decay_coef=1.0)
        mm = chip_metrics(g, c, float(sub["close"].iloc[-1]))
        conc = mm["concentration"]
        rows.append({
            "报告期": d.date(), "户数环比%": r["户数环比%"], "户均持股": int(round(r["户均持股"])),
            "集中度": round(conc, 3), "集中度环比": round(conc - prev, 3) if prev is not None else np.nan,
            "获利比例%": round(mm["profit_ratio"] * 100, 1),
        })
        prev = conc
    print(pd.DataFrame(rows).to_string(index=False))

    # ---------- 5. 关键筹码价位（套牢密集区 / 支撑） ----------
    g, c, m = chips_by_decay[1.0]
    order = np.argsort(c)[::-1]
    top = [(float(g[i]), float(c[i])) for i in order[:60]]
    above = sorted([(p, w) for p, w in top if p > cur], key=lambda x: -x[1])[:3]
    below = sorted([(p, w) for p, w in top if p <= cur], key=lambda x: -x[1])[:3]
    print("\n" + "=" * 86)
    print("  关键筹码价位（decay=1.0）")
    print("=" * 86)
    print(f"  上方套牢密集(压力): " + ", ".join(f"{p:.2f}({w*100:.1f}%)" for p, w in above))
    print(f"  下方筹码支撑      : " + ", ".join(f"{p:.2f}({w*100:.1f}%)" for p, w in below))

    # ---------- 6. 多衰减叠加图 ----------
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = {0.6: "#1f77b4", 1.0: "#000000", 1.5: "#d62728"}
    for dc in (0.6, 1.0, 1.5):
        gg, cc, _ = chips_by_decay[dc]
        ax.plot(cc, gg, color=colors[dc], lw=1.4, label=f"decay={dc}")
    ax.axhline(cur, color="#555", ls="--", lw=1.2)
    ax.text(ax.get_xlim()[1], cur, f" 现价 {cur:.2f}", va="bottom", ha="right", color="#555")
    ax.set_title(f"{CODE} {NAME} 不同衰减系数筹码分布对比")
    ax.set_xlabel("筹码占比"); ax.set_ylabel("价格"); ax.legend()
    plt.tight_layout(); plt.savefig("chip_002009_decay.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    # 同时刷新单图
    plot_chip_distribution(g, c, cur, m, title=f"{CODE} {NAME} 筹码分布", savepath="chip_002009.png")
    print("\n已出图: chip_002009.png, chip_002009_decay.png")


if __name__ == "__main__":
    main()
