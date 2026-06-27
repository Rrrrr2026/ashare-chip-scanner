# A股科技股板块筹码扫描器 (techscan)

一个本地交互式仪表盘：对申万**电子 + 计算机 + 通信**全板块（约 938 只）做「**股东户数 + 筹码分布**」双维扫描，可调参数、自由筛选/排序、互动图表。

> ⚠️ **免责声明**：筹码分布为历史换手**复刻估算**（非券商真实成本）；股东户数为季频公开数据；本工具仅作结构分析，**不构成任何投资建议**。

## 功能
- 🧭 **四象限互动图**：户数环比 × 溢价，点圆球看个股 K线 + 详情
- 📊 **结果表**：TradingView 式任意列区间筛选 + 多列优先级排序 + 一键预设方案
- 🔬 **个股筹码详情**：筹码分布图（含「**活筹/死筹**」两速锁仓模型，缓解低位筹码被低估的偏差）
- 🏛️ **机构足迹**：十大流通股东（含机构性质/增减）+ 北向持股历史
- 🔥 **各行业/上市板冷热**：行业、主板/创业板/科创板对比
- 💰 **成本低于现价分布**：浮盈安全垫分桶（区间宽度可调）

## 本地运行（推荐，功能完整）
```bash
git clone <你的仓库地址>
cd techscan
pip install -r requirements.txt
streamlit run techscan_app.py        # 或 Windows 双击 start.bat
```
浏览器自动打开 `http://localhost:8501`。仓库已带缓存数据（`techscan_data/`），开箱即用。

**更新/扩充数据**（联网，可断点续跑）：
```bash
python techscan_build.py             # 抓取并缓存；编辑 techscan_universe.csv 可改板块范围
```

## 部署到 Streamlit Community Cloud（任意电脑开网址即用）
1. 把本仓库推到 GitHub。
2. 登录 https://streamlit.io/cloud → New app → 选本仓库 → 主文件填 `techscan_app.py` → Deploy。
3. 得到一个公网 URL，任何电脑打开即可。

> ⚠️ **重大限制**：本应用实时取数依赖 akshare 抓取**国内数据源（东财/新浪/百度）**。Streamlit Cloud 服务器在**海外**，访问这些站点常**超时/不可达** —— 云端部署后**实时功能（个股 K线、机构足迹、重新抓数）大概率失效，只有仓库内缓存的静态数据能正常展示**。要完整功能请用「本地运行」。

## 数据来源
日线/换手 → 新浪　·　股东户数/总市值 → 东方财富数据中心　·　PE(TTM)/PB → 百度　·　行业成分 → 申万宏源官方　·　机构足迹 → 东财（十大股东 / 北向）

## 文件
| 文件 | 说明 |
|---|---|
| `techscan_app.py` | Streamlit 仪表盘主程序 |
| `techscan_build.py` | 数据构建器（联网、断点续跑） |
| `techscan_core.py` | 评分/象限/估值取数核心 |
| `chip_distribution.py` | 筹码分布模型（逐日换手迭代 + 活筹/死筹两速） |
| `holder_concentration.py` | 股东户数吸筹评分 |
| `techscan_universe.csv` | 板块成分（申万官方，可编辑） |
| `techscan_data/*.parquet` | 缓存数据（metrics + prices） |
| `test_chip.py` / `test_holder.py` | 离线单元测试 |
