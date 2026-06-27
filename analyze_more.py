# -*- coding: utf-8 -*-
"""① 核查 002009 户数+94% 的股本变动真相；② 多票横向扫描(户数+筹码)排序。"""
import numpy as np
import pandas as pd
import akshare as ak

from chip_distribution import fetch_hist, compute_chip_distribution, chip_metrics
from holder_concentration import fetch_holder_history, analyze_holder_history, _retry

pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 240)

# ---------------------------------------------------------------- #
# ① 002009 股本变动核查
# ---------------------------------------------------------------- #
def check_capital_change(code="002009"):
    print("=" * 90)
    print(f"  ① {code} 股东户数暴增核查：是不是股本变动/送转/解禁？")
    print("=" * 90)
    raw = _retry(lambda: ak.stock_zh_a_gdhs_detail_em(symbol=code))
    cols = list(raw.columns)
    # 容错挑列
    def pick(*kws, exclude=()):
        for c in cols:
            n = str(c)
            if all(k in n for k in kws) and not any(e in n for e in exclude):
                return c
        return None
    dcol = pick("截止日") or pick("截止")
    hcol = pick("股东户数", "本次") or pick("股东户数", exclude=["增减", "比例", "上次"])
    avg_sh = pick("户均持股数量") or pick("户均持股", "数量")
    tot = pick("总股本")
    chg = pick("股本变动")
    rsn = pick("股本变动原因")

    sub = raw[[c for c in [dcol, hcol, avg_sh, tot, chg, rsn] if c]].copy()
    sub = sub.rename(columns={dcol: "截止日", hcol: "股东户数"})
    sub["截止日"] = pd.to_datetime(sub["截止日"])
    sub = sub.sort_values("截止日").tail(6)
    print(sub.to_string(index=False))
    print("\n解读：户均持股数量若同步跳变=有股本变动(送转)；若总股本/股本变动有数且原因含解禁/定增=供给事件。")
    print("      若户数翻倍而总股本基本不变=真实新股东涌入(追高/接盘)，而非锁仓。\n")


# ---------------------------------------------------------------- #
# ② 多票横向扫描：户数维度 + 筹码维度
# ---------------------------------------------------------------- #
WATCH = {
    "002009": "天奇股份", "300274": "阳光电源", "600519": "贵州茅台",
    "000001": "平安银行", "300750": "宁德时代",
}


def scan_combined(watch=WATCH):
    print("=" * 90)
    print("  ② 横向扫描（户数吸筹分 + 当前筹码），按吸筹分降序")
    print("=" * 90)
    rows = []
    for code, name in watch.items():
        rec = {"代码": code, "名称": name}
        try:
            h = analyze_holder_history(fetch_holder_history(code))
            last = h.iloc[-1]
            rec.update({
                "截止日": pd.to_datetime(last["date"]).date(),
                "户数环比%": last["户数环比%"],
                "标签": last["标签"],
                "吸筹分": last["吸筹分"],
            })
        except Exception as e:
            rec.update({"截止日": None, "户数环比%": None, "标签": f"户数失败:{type(e).__name__}", "吸筹分": np.nan})
        try:
            df = fetch_hist(code, start_date="20220101")
            cur = float(df["close"].iloc[-1])
            g, c = compute_chip_distribution(df, decay_coef=1.0)
            m = chip_metrics(g, c, cur)
            rec.update({
                "现价": round(cur, 2),
                "获利比例%": round(m["profit_ratio"] * 100, 1),
                "集中度": round(m["concentration"], 3),
            })
        except Exception as e:
            rec.update({"现价": None, "获利比例%": None, "集中度": f"筹码失败:{type(e).__name__}"})
        rows.append(rec)

    df = pd.DataFrame(rows)[
        ["代码", "名称", "截止日", "户数环比%", "标签", "吸筹分", "现价", "获利比例%", "集中度"]
    ]
    df = df.sort_values("吸筹分", ascending=False, na_position="last").reset_index(drop=True)
    print(df.to_string(index=False))
    print("\n双重锁仓优选 = 户数环比<0 且 吸筹分高 且 集中度小：")
    good = df[(pd.to_numeric(df["户数环比%"], errors="coerce") < 0) &
             (pd.to_numeric(df["吸筹分"], errors="coerce") >= 60)]
    print("  ", good["代码"].tolist() if len(good) else "（本批无）")


if __name__ == "__main__":
    check_capital_change("002009")
    scan_combined()
    print("\n完成。")
