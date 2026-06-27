# -*- coding: utf-8 -*-
"""
test_holder.py — 工具二(holder_concentration) 离线自测，纯本地 mock 数据，不联网。

1. 造一段 mock 股东户数历史（含持续下降 + 一段回升），跑 analyze_holder_history 打印表。
2. 对 label_period / holder_score 做边界自检，覆盖 5 种情景并 assert 标签 / 分数正确。
"""

import numpy as np
import pandas as pd

from holder_concentration import (
    label_period,
    holder_score,
    analyze_holder_history,
)


def make_mock_history() -> pd.DataFrame:
    """
    mock 一段股东户数历史：先持续下降（筹码集中），后回升（散户回流）。
    price_chg 为该统计区间的股价涨跌幅（小数）。
    """
    dates = pd.to_datetime(
        [
            "2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30",
            "2025-05-31", "2025-06-30", "2025-07-31", "2025-08-31",
        ]
    )
    holders = [100000, 96000, 91000, 86000, 80000, 84000, 90000, 97000]
    #          ----------- 持续下降 -----------   ----- 回升 -----
    price_chg = [0.00, 0.01, 0.03, 0.08, 0.20, -0.06, 0.12, 0.10]
    return pd.DataFrame({"date": dates, "holders": holders, "price_chg": price_chg})


def test_history():
    pd.set_option("display.unicode.east_asian_width", True)
    df = make_mock_history()
    res = analyze_holder_history(df)
    show = res[["date", "holders", "户数环比%", "涨跌幅%", "标签", "吸筹分"]]
    print("=== mock 股东户数历史分析 ===")
    print(show.to_string(index=False))
    print()


def test_label_boundaries():
    """5 种情景的边界自检。"""
    cases = []

    # 1) 户数大降 + 价平 -> "低位吸筹"，高分
    s, lab, _ = holder_score(-0.05, 0.00)
    print(f"[情景1] 户数-5% 价0%   -> {lab:<22} 分={s:.1f}")
    assert lab == "低位吸筹", lab
    assert s >= 80, f"低位吸筹应为高分, got {s}"
    cases.append(lab)

    # 2) 户数降 + 已大涨 -> "高位集中(已大涨,谨慎)"，中段分
    s, lab, _ = holder_score(-0.05, 0.40)
    print(f"[情景2] 户数-5% 价+40% -> {lab:<22} 分={s:.1f}")
    assert lab == "高位集中(已大涨,谨慎)", lab
    assert 45 <= s <= 65, f"高位集中应为中段分, got {s}"
    cases.append(lab)

    # 3) 户数大增 + 价跌 -> "派发/散户接盘"，接近 0
    s, lab, _ = holder_score(0.08, -0.10)
    print(f"[情景3] 户数+8% 价-10% -> {lab:<22} 分={s:.1f}")
    assert lab == "派发/散户接盘", lab
    assert s <= 5, f"派发应接近 0, got {s}"
    cases.append(lab)

    # 4) 户数升 + 价涨 -> "追高进场"
    s, lab, _ = holder_score(0.05, 0.10)
    print(f"[情景4] 户数+5% 价+10% -> {lab:<22} 分={s:.1f}")
    assert lab == "追高进场", lab
    cases.append(lab)

    # 5) 几乎没变 -> "中性"，约 50 分
    s, lab, _ = holder_score(0.0, 0.0)
    print(f"[情景5] 户数 0% 价 0%   -> {lab:<22} 分={s:.1f}")
    assert lab == "中性", lab
    assert abs(s - 50) < 1e-6, f"中性应约 50 分, got {s}"
    cases.append(lab)

    # 直接对 label_period 再确认一遍边界
    assert label_period(-0.03, 0.05)[0] == "低位吸筹"
    assert label_period(-0.03, 0.10)[0] == "拉升中集中"
    assert label_period(-0.03, 0.30)[0] == "高位集中(已大涨,谨慎)"
    assert label_period(0.03, -0.01)[0] == "派发/散户接盘"
    assert label_period(0.03, 0.01)[0] == "追高进场"
    assert label_period(0.0, 0.5)[0] == "中性"

    print("\n[断言通过] 5 种情景标签与分数均正确")


def main():
    test_history()
    test_label_boundaries()
    print("\n全部通过")


if __name__ == "__main__":
    main()
