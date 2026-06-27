# -*- coding: utf-8 -*-
"""
锂电池回收 / 汽车拆解·再生资源 板块全成分双维扫描。
成分表为人工锁定（东财板块成分接口此刻被限频）；解封后可用
ak.stock_board_concept_cons_em(symbol='锂电池回收') 替换 CONSTITUENTS。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from chip_distribution import (
    fetch_hist, compute_chip_distribution, chip_metrics, setup_chinese_font,
)
from holder_concentration import fetch_holder_history, analyze_holder_history

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 300)

# 人工锁定核心成分（锂电回收 + 再生金属 + 汽车拆解/再生资源）
CONSTITUENTS = {
    "002340": "格林美", "603799": "华友钴业", "300919": "中伟股份",
    "002741": "光华科技", "300409": "道氏技术", "300618": "寒锐钴业",
    "688779": "长远锂科", "300068": "南都电源", "601311": "骆驼股份",
    "002009": "天奇股份", "002645": "华宏科技", "601388": "怡球资源",
    "600217": "中再资环", "002034": "旺能环境", "002672": "东江环保",
    "002266": "浙富控股", "603588": "高能环境", "000546": "金圆股份",
    "301049": "超越科技", "600961": "株冶集团", "002460": "赣锋锂业",
    "002466": "天齐锂业",
}


def chip_clean_score(profit, conc, d_conc, premium):
    s = 0
    if not np.isnan(d_conc) and d_conc < 0:
        s += 15
    if conc < 0.25:
        s += 25
    elif conc < 0.35:
        s += 12
    if 40 <= profit <= 88:
        s += 30
    elif profit > 88:
        s += 8
    if premium >= 0:
        s += 30
    elif premium >= -8:
        s += 12
    return s


def conc_at(df, date):
    sub = df[df["date"] <= pd.to_datetime(date)]
    if len(sub) < 30:
        return np.nan
    g, c = compute_chip_distribution(sub, decay_coef=1.0)
    return chip_metrics(g, c, float(sub["close"].iloc[-1]))["concentration"]


def quadrant(holder_chg, premium):
    """按 户数环比 / 溢价 归四象限。"""
    if holder_chg is None or np.isnan(holder_chg):
        return "数据缺失"
    down = holder_chg < 0
    up_cost = premium >= 0
    if down and up_cost:
        return "强势控盘"        # 户数降+浮盈
    if down and not up_cost:
        return "低位吸筹/筑底"    # 户数降+套牢
    if (not down) and up_cost:
        return "高位追高派发"     # 户数升+浮盈
    return "套牢派发"            # 户数升+套牢


def main():
    rows = []
    for code, name in CONSTITUENTS.items():
        rec = {"代码": code, "名称": name}
        # 户数维度
        try:
            h = analyze_holder_history(fetch_holder_history(code))
            last = h.iloc[-1]
            rec["截止日"] = pd.to_datetime(last["date"]).date()
            rec["户数环比%"] = last["户数环比%"]
            rec["吸筹分"] = last["吸筹分"]
            rec["标签"] = last["标签"]
            dates = list(h["date"].tail(2))
        except Exception as e:
            rec.update({"截止日": None, "户数环比%": np.nan, "吸筹分": np.nan,
                        "标签": f"户数失败:{type(e).__name__}"})
            dates = []
        # 筹码维度（直接用新浪源，跳过被封的东财，提速）
        try:
            df = fetch_hist(code, start_date="20220101", source="sina")
            cur = float(df["close"].iloc[-1])
            g, c = compute_chip_distribution(df, decay_coef=1.0)
            m = chip_metrics(g, c, cur)
            profit = m["profit_ratio"] * 100
            conc = m["concentration"]
            premium = (cur / m["avg_cost"] - 1) * 100
            d_conc = np.nan
            if len(dates) == 2:
                cp, cn = conc_at(df, dates[0]), conc_at(df, dates[1])
                if not (np.isnan(cp) or np.isnan(cn)):
                    d_conc = cn - cp
            rec.update({
                "现价": round(cur, 2), "溢价%": round(premium, 1),
                "获利%": round(profit, 1), "集中度": round(conc, 3),
                "集中度环比": round(d_conc, 3) if not np.isnan(d_conc) else np.nan,
                "主峰": round(m["peak_price"], 2),
                "_chip": chip_clean_score(profit, conc, d_conc, premium),
                "_premium": premium,
            })
        except Exception as e:
            rec.update({"现价": np.nan, "溢价%": np.nan, "获利%": np.nan,
                        "集中度": np.nan, "_chip": np.nan, "_premium": np.nan,
                        "标签": rec.get("标签", "") + f"|筹码失败:{type(e).__name__}"})
        rec["象限"] = quadrant(rec.get("户数环比%"), rec.get("_premium", np.nan))
        rows.append(rec)
        print(f"  done {code} {name}")

    df = pd.DataFrame(rows)
    df["筹码分"] = df["_chip"]
    df["综合分"] = (0.5 * pd.to_numeric(df["吸筹分"], errors="coerce")
                    + 0.5 * pd.to_numeric(df["筹码分"], errors="coerce")).round(1)
    df = df.sort_values("综合分", ascending=False, na_position="last").reset_index(drop=True)
    df.to_csv("sector_scan.csv", index=False, encoding="utf-8-sig")

    cols = ["代码", "名称", "户数环比%", "吸筹分", "现价", "溢价%", "获利%",
            "集中度", "集中度环比", "筹码分", "综合分", "象限", "标签"]
    print("\n" + "=" * 120)
    print("  锂电回收/再生资源 板块双维扫描（综合分降序）")
    print("=" * 120)
    print(df[cols].to_string(index=False))

    # 四象限分组
    print("\n【四象限归类】")
    for q in ["强势控盘", "低位吸筹/筑底", "高位追高派发", "套牢派发", "数据缺失"]:
        sub = df[df["象限"] == q]
        if len(sub):
            print(f"  {q}: " + ", ".join(f"{r['名称']}({r['代码']})" for _, r in sub.iterrows()))

    # 四象限散点图
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(12, 9))
    d = df.dropna(subset=["户数环比%", "溢价%"])
    sc = ax.scatter(d["户数环比%"], d["溢价%"],
                    s=80 + 4 * pd.to_numeric(d["综合分"], errors="coerce").fillna(0),
                    c=pd.to_numeric(d["综合分"], errors="coerce"), cmap="RdYlGn",
                    edgecolors="k", linewidths=0.6, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(r["名称"], (r["户数环比%"], r["溢价%"]),
                    fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="#666", lw=1); ax.axvline(0, color="#666", lw=1)
    ax.set_xlabel("户数环比 %  (← 左=户数降/筹码集中)")
    ax.set_ylabel("溢价 % = 现价/平均成本-1  (↑ 上=浮盈)")
    ax.set_title("锂电回收/再生资源 板块筹码四象限\n左上=强势控盘 左下=低位吸筹 右上=高位追高 右下=套牢派发")
    plt.colorbar(sc, label="综合分")
    # 象限标注
    xl, xr = ax.get_xlim(); yb, yt = ax.get_ylim()
    ax.text(xl*0.95, yt*0.9, "强势控盘", color="green", fontsize=11, alpha=0.5)
    ax.text(xl*0.95, yb*0.9, "低位吸筹/筑底", color="blue", fontsize=11, alpha=0.5)
    ax.text(xr*0.4, yt*0.9, "高位追高派发", color="orange", fontsize=11, alpha=0.5)
    ax.text(xr*0.4, yb*0.9, "套牢派发", color="red", fontsize=11, alpha=0.5)
    plt.tight_layout(); plt.savefig("sector_quadrant.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("\n已出图: sector_quadrant.png | 明细已存 sector_scan.csv")


if __name__ == "__main__":
    main()
