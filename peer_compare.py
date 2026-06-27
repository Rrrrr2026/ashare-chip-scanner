# -*- coding: utf-8 -*-
"""002009 锂电回收/汽车拆解 同行横向对比：户数维度 + 筹码维度，找筹码更干净的票。"""
import numpy as np
import pandas as pd

from chip_distribution import fetch_hist, compute_chip_distribution, chip_metrics
from holder_concentration import fetch_holder_history, analyze_holder_history

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 260)

PEERS = {
    "002009": "天奇股份", "002340": "格林美", "002741": "光华科技",
    "300409": "道氏技术", "603799": "华友钴业", "002645": "华宏科技",
    "002034": "旺能环境",
}


def chip_clean_score(profit, conc, d_conc, premium):
    """筹码干净度 0~100：户数维度之外，单看筹码结构是否像健康吸筹。"""
    s = 0
    if not np.isnan(d_conc) and d_conc < 0:      # 集中度在收窄
        s += 15
    if conc < 0.25:                               # 筹码高度锁定
        s += 25
    elif conc < 0.35:
        s += 12
    if 40 <= profit <= 88:                         # 主力浮盈、不深套也不过热
        s += 30
    elif profit > 88:
        s += 8
    if premium >= 0:                               # 现价不低于平均成本(整体不亏)
        s += 30
    elif premium >= -8:
        s += 12
    return s


def conc_at(df, date, decay=1.0):
    sub = df[df["date"] <= pd.to_datetime(date)]
    if len(sub) < 30:
        return np.nan
    g, c = compute_chip_distribution(sub, decay_coef=decay)
    return chip_metrics(g, c, float(sub["close"].iloc[-1]))["concentration"]


def main():
    rows = []
    for code, name in PEERS.items():
        rec = {"代码": code, "名称": name}
        try:
            h = analyze_holder_history(fetch_holder_history(code))
            last = h.iloc[-1]
            rec["户数环比%"] = last["户数环比%"]
            rec["吸筹分"] = last["吸筹分"]
            rec["标签"] = last["标签"]
            dates = list(h["date"].tail(2))
        except Exception as e:
            rec.update({"户数环比%": None, "吸筹分": np.nan, "标签": f"户数失败:{type(e).__name__}"})
            dates = []
        try:
            df = fetch_hist(code, start_date="20220101")
            cur = float(df["close"].iloc[-1])
            g, c = compute_chip_distribution(df, decay_coef=1.0)
            m = chip_metrics(g, c, cur)
            profit = m["profit_ratio"] * 100
            conc = m["concentration"]
            premium = (cur / m["avg_cost"] - 1) * 100
            d_conc = np.nan
            if len(dates) == 2:
                c_prev, c_now = conc_at(df, dates[0]), conc_at(df, dates[1])
                if not (np.isnan(c_prev) or np.isnan(c_now)):
                    d_conc = c_now - c_prev
            rec.update({
                "现价": round(cur, 2), "平均成本": round(m["avg_cost"], 2),
                "溢价%": round(premium, 1), "获利%": round(profit, 1),
                "集中度": round(conc, 3), "集中度环比": round(d_conc, 3) if not np.isnan(d_conc) else np.nan,
                "主峰": round(m["peak_price"], 2),
                "_chip": chip_clean_score(profit, conc, d_conc, premium),
            })
        except Exception as e:
            rec.update({"现价": None, "_chip": np.nan, "集中度": f"失败:{type(e).__name__}"})
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["筹码分"] = df["_chip"]
    df["综合分"] = (0.5 * pd.to_numeric(df["吸筹分"], errors="coerce")
                    + 0.5 * pd.to_numeric(df["筹码分"], errors="coerce")).round(1)
    df = df.sort_values("综合分", ascending=False, na_position="last").reset_index(drop=True)

    cols = ["代码", "名称", "户数环比%", "吸筹分", "标签", "现价", "平均成本",
            "溢价%", "获利%", "集中度", "集中度环比", "主峰", "筹码分", "综合分"]
    print("=" * 110)
    print("  锂电回收 / 汽车拆解 同行横向对比（综合分 = 0.5×吸筹分[户数] + 0.5×筹码分[结构]）")
    print("=" * 110)
    print(df[cols].to_string(index=False))

    print("\n【筛选】户数在降 + 综合分领先 = 筹码更干净候选：")
    good = df[(pd.to_numeric(df["户数环比%"], errors="coerce") < 0)].head(3)
    for _, r in good.iterrows():
        print(f"  {r['代码']} {r['名称']}: 综合{r['综合分']} | 户数{r['户数环比%']}% 吸筹{r['吸筹分']} | "
              f"获利{r['获利%']}% 集中度{r['集中度']} 溢价{r['溢价%']}%")


if __name__ == "__main__":
    main()
