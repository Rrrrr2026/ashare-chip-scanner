# -*- coding: utf-8 -*-
"""
test_chip.py — 工具一(chip_distribution) 离线自测，纯本地合成数据，不联网。

合成 ~400 根日线，分三段：
    低位横盘吸筹(低换手 0.01~0.03) -> 放量拉升(换手 0.05~0.12) -> 高位回落(换手 0.03~0.06)
对 decay_coef ∈ {0.6, 1.0, 1.5} 跑 chip_metrics 对比，验证：
    衰减系数越大 -> 平均成本越低、集中度越小。
并用 decay=1.0 出一张 chip_demo.png 验证画图 + 中文字体。
"""

import numpy as np
import pandas as pd

from chip_distribution import (
    compute_chip_distribution,
    chip_metrics,
    plot_chip_distribution,
)


def make_synthetic(seed: int = 42) -> pd.DataFrame:
    """造三段式合成日线：吸筹 -> 拉升 -> 回落。"""
    rng = np.random.default_rng(seed)

    n_acc, n_up, n_down = 150, 100, 150  # 共 400 根

    # 段一：低位横盘吸筹，价格在 10 附近小幅波动
    acc = 10.0 + rng.normal(0, 0.15, n_acc).cumsum() * 0.05
    acc = np.clip(acc, 9.3, 10.7)
    acc_turn = rng.uniform(0.01, 0.03, n_acc)

    # 段二：放量拉升，10 -> 30
    up = np.linspace(acc[-1], 30.0, n_up) + rng.normal(0, 0.25, n_up)
    up_turn = rng.uniform(0.05, 0.12, n_up)

    # 段三：高位回落，30 -> 21
    down = np.linspace(up[-1], 21.0, n_down) + rng.normal(0, 0.3, n_down)
    down_turn = rng.uniform(0.03, 0.06, n_down)

    close = np.concatenate([acc, up, down])
    turnover = np.concatenate([acc_turn, up_turn, down_turn])

    # high/low 在 close 上下加小幅噪声
    spread = np.abs(rng.normal(0, 0.008, close.size)) + 0.004
    high = close * (1 + spread)
    low = close * (1 - spread)

    dates = pd.bdate_range("2023-01-03", periods=close.size)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "high": high,
            "low": low,
            "turnover": turnover,
        }
    )


def main():
    pd.set_option("display.unicode.east_asian_width", True)

    df = make_synthetic()
    current = float(df["close"].iloc[-1])
    print(f"合成日线 {len(df)} 根，现价(末日收盘) = {current:.2f}\n")

    rows = []
    metrics_by_decay = {}
    for decay in (0.6, 1.0, 1.5):
        grid, chips = compute_chip_distribution(df, decay_coef=decay)
        m = chip_metrics(grid, chips, current)
        metrics_by_decay[decay] = m
        rows.append(
            {
                "衰减系数": decay,
                "平均成本": round(m["avg_cost"], 3),
                "主峰价位": round(m["peak_price"], 3),
                "获利比例%": round(m["profit_ratio"] * 100, 1),
                "集中度": round(m["concentration"], 4),
                "90%下沿": round(m["cost_low"], 2),
                "90%上沿": round(m["cost_high"], 2),
            }
        )

    table = pd.DataFrame(rows)
    print("=== 不同衰减系数下的筹码指标对比 ===")
    print(table.to_string(index=False))
    print()

    avg = {d: metrics_by_decay[d]["avg_cost"] for d in (0.6, 1.0, 1.5)}
    con = {d: metrics_by_decay[d]["concentration"] for d in (0.6, 1.0, 1.5)}

    print(f"平均成本: 0.6 -> {avg[0.6]:.3f} | 1.0 -> {avg[1.0]:.3f} | 1.5 -> {avg[1.5]:.3f}")
    print(f"集中度  : 0.6 -> {con[0.6]:.4f} | 1.0 -> {con[1.0]:.4f} | 1.5 -> {con[1.5]:.4f}")

    # 验证：衰减系数越大 -> 平均成本越低（更看重近期回落后的较低成本）
    assert avg[0.6] > avg[1.0] > avg[1.5], "断言失败：平均成本应随衰减系数增大而降低"
    # 验证：衰减系数越大 -> 集中度越小（分布更贴近近期、更窄）
    assert con[0.6] > con[1.0] > con[1.5], "断言失败：集中度应随衰减系数增大而减小"
    print("\n[断言通过] 衰减系数越大 -> 平均成本越低、集中度越小")

    # decay=1.0 出图，验证画图与中文字体
    grid, chips = compute_chip_distribution(df, decay_coef=1.0)
    m = chip_metrics(grid, chips, current)
    plot_chip_distribution(
        grid, chips, current, m,
        title="合成数据 筹码分布 (decay=1.0)",
        savepath="chip_demo.png",
    )
    import os
    assert os.path.exists("chip_demo.png"), "断言失败：chip_demo.png 未生成"
    print("[出图通过] 已生成 chip_demo.png")

    print("\n全部通过")


if __name__ == "__main__":
    main()
