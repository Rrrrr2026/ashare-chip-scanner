# 科技股板块筹码扫描器 (techscan)

本地交互式仪表盘：全板块"户数 + 筹码"双维扫描，参数自由调、结果自由排序/筛选、炫酷互动图表。

## 一键启动

双击 **`run_techscan.bat`**（首次会自动抓数据，之后直接开仪表盘）。
或手动：

```bash
python techscan_build.py        # 1) 抓数据 + 预算 5 档衰减系数指标，缓存到 techscan_data/
streamlit run techscan_app.py   # 2) 打开仪表盘（默认 http://localhost:8501）
```

## 设定"板块范围"

编辑 **`techscan_universe.csv`**（两列 `code,name`），加你要扫的票，然后：

```bash
python techscan_build.py        # 断点续跑：已抓过的自动跳过，只补新增的
python techscan_build.py --refresh   # 全量重抓
```

> 想扫"整个科技股板块"（几百只）：把全部代码填进 csv 即可。建议分批/挂后台跑——
> akshare 数据源限频，串行+重试+新浪回落才能抓全；断点续跑保证中断了能接着来。

## 仪表盘能干什么

**侧边栏（实时调参，秒响应）**
- 衰减系数 decay（5 档）、评分权重（户数 vs 筹码）
- 筹码评分阈值：集中度锁仓线、获利健康区间
- 象限分界线：户数环比分界、溢价分界
- 筛选：象限多选、PE(TTM) 区间、综合分下限

**三个页签**
1. 🧭 **四象限互动图** — 悬停看详情/框选缩放/双击复位，颜色+大小=综合分
2. 📊 **结果表** — 点列名或用下拉**任意字段排序**（溢价/PE/综合分/集中度…），一键导出 CSV
3. 🔬 **个股筹码详情** — 选股看该票筹码分布图（按当前衰减档实时算）+ 全套指标

## 数据源（已绕开东财限频）
- 日线/换手 → 新浪源　·　股东户数 → 东财数据中心　·　PE(TTM)/总市值(亿)/PB → 百度源
- 筹码为历史换手迭代**复刻**（非券商真实成本），仅供结构参考

## 文件
- `techscan_core.py` — 纯计算/取数核心（评分、象限、估值取数）
- `techscan_build.py` — 数据构建器（联网、断点续跑）
- `techscan_app.py` — Streamlit 仪表盘
- `techscan_universe.csv` — 板块范围（可编辑）
- `techscan_data/` — 缓存（metrics.parquet + prices.parquet）
