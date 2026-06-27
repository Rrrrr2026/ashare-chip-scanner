# -*- coding: utf-8 -*-
"""
techscan_app.py — 科技股板块筹码扫描器 · 交互式仪表盘（Streamlit + Plotly）。

运行：  streamlit run techscan_app.py
功能：  调参（衰减系数/评分权重/阈值/象限分界）+ 自由排序 + 多维筛选 + 互动四象限图 + 个股筹码详情。
数据：  读 techscan_data/*.parquet（先跑 python techscan_build.py 生成）。
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from techscan_core import (
    DECAY_LABELS, QUADRANTS, QUAD_COLORS, derive_view,
)
from chip_distribution import compute_chip_distribution, chip_metrics, fetch_hist, _sina_symbol
from holder_concentration import fetch_holder_history, analyze_holder_history

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "techscan_data")
METRICS_PARQUET = os.path.join(DATA_DIR, "metrics.parquet")
PRICES_PARQUET = os.path.join(DATA_DIR, "prices.parquet")
UNIVERSE_CSV = os.path.join(HERE, "techscan_universe.csv")

st.set_page_config(page_title="科技股板块筹码扫描器", layout="wide", page_icon="📈")


def board_of(code):
    """按代码前缀判定上市板：主板 / 创业板 / 科创板 / 北交所。"""
    c = str(code).zfill(6)
    if c[:3] in ("688", "689"):
        return "科创板"
    if c[:3] in ("300", "301"):
        return "创业板"
    if c[0] in ("4", "8") or c[:3] == "920":
        return "北交所"
    if c[0] in ("6", "0"):
        return "主板"
    return "其他"


# 一键预设（在结果表内套用筛选+排序；行业/上市板侧栏筛选仍生效，象限由预设指定）
PRESETS = {
    "（手动筛选）": None,
    "🎯 便宜+控盘 · 严格 (PE 0~40 · 强势控盘 · 集中度<0.25)":
        {"quad": ["强势控盘"], "num": {"pe_ttm": (0, 40), "集中度": (0, 0.25)}, "sort": ["pe_ttm"], "asc": True},
    "🎯 便宜+控盘 · 放宽 (PE 0~40 · 强势控盘 · 集中度<0.35)":
        {"quad": ["强势控盘"], "num": {"pe_ttm": (0, 40), "集中度": (0, 0.35)}, "sort": ["pe_ttm"], "asc": True},
    "🛡️ 低位吸筹+锁仓 (低位吸筹/筑底 · 集中度<0.30)":
        {"quad": ["低位吸筹/筑底"], "num": {"集中度": (0, 0.30)}, "sort": ["综合分"], "asc": False},
    "💎 深安全垫 (浮盈 · 溢价>20%)":
        {"num": {"溢价%": (20, 1e9)}, "sort": ["溢价%"], "asc": False},
}


@st.cache_data(show_spinner=False)
def load_data(mtime=None):
    if not os.path.exists(METRICS_PARQUET):
        return None, None
    metrics = pd.read_parquet(METRICS_PARQUET)
    # 合入申万行业标签（来自 universe.csv）
    if os.path.exists(UNIVERSE_CSV):
        uni = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
        uni["code"] = uni["code"].str.zfill(6)
        if "industry" in uni.columns:
            metrics = metrics.merge(
                uni[["code", "industry"]].rename(columns={"code": "代码", "industry": "行业"}),
                on="代码", how="left",
            )
    if "行业" not in metrics.columns:
        metrics["行业"] = "—"
    metrics["行业"] = metrics["行业"].fillna("—")
    metrics["上市板"] = metrics["代码"].map(board_of)  # 主板/创业板/科创板/北交所
    prices = pd.read_parquet(PRICES_PARQUET) if os.path.exists(PRICES_PARQUET) else pd.DataFrame()
    return metrics, prices


_mtime = os.path.getmtime(METRICS_PARQUET) if os.path.exists(METRICS_PARQUET) else None
metrics, prices = load_data(_mtime)

st.title("📈 科技股板块筹码扫描器")
if metrics is None or not len(metrics):
    st.error("还没有数据。请先运行：`python techscan_build.py` 生成缓存（可在 techscan_universe.csv 里设定板块范围）。")
    st.stop()

st.caption(
    f"共 {len(metrics)} 只 | 数据截止 {pd.to_datetime(metrics['末日']).max().date() if '末日' in metrics else '—'} "
    "| 筹码为历史换手迭代复刻（非真实成本）；估值=百度源 PE(TTM)/总市值(亿)/PB"
)

# ----------------------------- 侧边栏：参数 + 筛选 ----------------------------- #
with st.sidebar:
    st.header("⚙️ 参数")
    decay = st.select_slider("衰减系数 decay", options=DECAY_LABELS, value="1.0",
                             help="<1 历史套牢盘更黏；>1 更看重近期成本")
    lock_strength = st.slider("锁仓强度（活筹/死筹）", 0.0, 1.0, 0.0, 0.1,
                              help="0=经典均匀换手；>0 把深度获利的低位筹码视为庄家/机构锁仓盘、少被换手清掉，"
                                   "缓解『低位筹码被低估』的偏差。影响『个股筹码详情』与点击预览的成本估算。")
    w_holder = st.slider("评分权重：户数维度占比", 0.0, 1.0, 0.5, 0.05,
                         help="综合分 = 该比例×吸筹分 + (1-比例)×筹码分")
    st.markdown("**筹码评分阈值**")
    conc_lock = st.slider("集中度『锁仓』阈值（越小越锁）", 0.10, 0.50, 0.25, 0.01)
    profit_lo, profit_hi = st.slider("获利比例『健康区间』%", 0, 100, (40, 88))
    st.markdown("**象限分界线**")
    h_thr = st.slider("户数环比分界 %（左=户数降）", -10.0, 10.0, 0.0, 0.5)
    p_thr = st.slider("溢价分界 %（上=浮盈）", -20.0, 20.0, 0.0, 1.0)

    st.divider()
    st.header("🔎 筛选")
    inds = sorted([x for x in metrics["行业"].dropna().unique().tolist() if x != "—"]) or ["—"]
    ind_sel = st.multiselect("行业", inds, default=inds)
    board_order = ["主板", "创业板", "科创板", "北交所", "其他"]
    boards = [b for b in board_order if b in set(metrics["上市板"])]
    board_sel = st.multiselect("上市板", boards, default=boards)
    quad_sel = st.multiselect("象限", QUADRANTS,
                              default=["强势控盘", "低位吸筹/筑底", "高位追高派发", "套牢派发"])

# 派生视图
view = derive_view(metrics, decay, w_holder, conc_lock, profit_lo, profit_hi, h_thr, p_thr)

# 行业 + 上市板筛选（不含象限）—— 供「行业冷热 / 成本分布」等全体概览页使用。
f_ind = view.copy()
f_ind = f_ind[f_ind["行业"].isin(ind_sel)] if ind_sel else f_ind
f_ind = f_ind[f_ind["上市板"].isin(board_sel)] if board_sel else f_ind
# f = 行业 + 象限筛选 —— 供四象限图 / KPI / 结果表使用。
# 数值区间筛选 + 多列排序放到「结果表」内，做成 TradingView 式（见 Tab2）。
f = f_ind.copy()
f = f[f["象限"].isin(quad_sel)] if quad_sel else f

# ----------------------------- 顶部 KPI ----------------------------- #
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("筛选后", f"{len(f)} 只")
for col, q in zip([c2, c3, c4, c5],
                  ["强势控盘", "低位吸筹/筑底", "高位追高派发", "套牢派发"]):
    col.metric(q, int((f["象限"] == q).sum()))

# ----------------------------- 个股明细：按需取数（缓存） ----------------------------- #
@st.cache_data(show_spinner=False)
def load_kline(code):
    df = fetch_hist(code, start_date="20210101", source="sina").sort_values("date")
    for w in (10, 30, 120, 250):
        df[f"MA{w}"] = df["close"].rolling(w).mean()
    return df


@st.cache_data(show_spinner=False)
def load_holder_recent(code, n=6):
    h = analyze_holder_history(fetch_holder_history(code))
    out = h[["date", "holders", "户数环比%", "涨跌幅%", "标签"]].tail(n).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.rename(columns={"date": "截止日", "holders": "股东户数"})
    return out


@st.cache_data(show_spinner=False)
def load_revenue(code, n=4):
    import akshare as ak
    q = ak.stock_financial_abstract_ths(symbol=code, indicator="按单季度")
    cols = [c for c in ["报告期", "营业总收入", "营业总收入同比增长率"] if c in q.columns]
    out = q[cols].tail(n).rename(columns={"营业总收入": "单季营收", "营业总收入同比增长率": "同比"})
    return out


@st.cache_data(show_spinner=False)
def load_top10(code):
    """十大流通股东（季频，含机构性质/增减）。尝试最近几个报告期。"""
    import akshare as ak
    sym = _sina_symbol(code)
    for dt in ["20260331", "20251231", "20250930", "20250630"]:
        try:
            d = ak.stock_gdfx_free_top_10_em(symbol=sym, date=dt)
            if d is not None and len(d):
                keep = [c for c in ["名次", "股东名称", "股东性质", "持股数", "占总流通股本持股比例",
                                    "增减", "变动比率"] if c in d.columns]
                return dt, d[keep]
        except Exception:
            pass
    return None, None


@st.cache_data(show_spinner=False)
def load_northbound(code):
    """北向(陆股通)个股持股历史。注:交易所自 2024-08 起停止披露个股日数据。"""
    import akshare as ak
    d = ak.stock_hsgt_individual_em(symbol=code)
    pcol = next((c for c in d.columns if "占A股百分比" in str(c) or "占" in str(c)), None)
    dcol = next((c for c in d.columns if "日期" in str(c)), None)
    return d, dcol, pcol


def render_stock_preview(code, key_prefix):
    if not code:
        st.info("👆 点击上方的圆球 / 表格行，这里会显示该股的 K 线预览与详情。")
        return
    row = view[view["代码"] == code]
    if not len(row):
        st.warning("该股不在当前数据中。")
        return
    r = row.iloc[0]
    st.markdown(f"#### 📌 {code} {r['名称']}　·　{r.get('行业', '')}　·　{r.get('上市板', '')}　|　{r.get('象限', '')}")

    with st.spinner("加载个股明细…"):
        # K 线 + 均线
        try:
            k = load_kline(code).tail(250)
            fig = go.Figure(go.Candlestick(
                x=k["date"], open=k["open"], high=k["high"], low=k["low"], close=k["close"],
                name="日K", increasing_line_color="#d62728", decreasing_line_color="#2ca02c"))
            for w, clr in [(10, "#1f77b4"), (30, "#ff7f0e"), (120, "#9467bd"), (250, "#8c564b")]:
                fig.add_trace(go.Scatter(x=k["date"], y=k[f"MA{w}"], mode="lines",
                                         name=f"MA{w}", line=dict(width=1.2, color=clr)))
            fig.update_layout(height=380, template="plotly_white", xaxis_rangeslider_visible=False,
                              margin=dict(l=10, r=10, t=10, b=10),
                              legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True, key=f"k_{key_prefix}_{code}")
        except Exception as e:
            st.warning(f"K 线获取失败：{type(e).__name__}")

        # 锁仓修正成本（用侧栏 lock_strength 现场重算该股筹码）
        cost_lock = None
        if lock_strength > 0:
            try:
                kf = load_kline(code)
                if "turnover" in kf.columns:
                    gg, cc = compute_chip_distribution(kf, decay_coef=float(decay),
                                                       lock_strength=lock_strength)
                    cost_lock = chip_metrics(gg, cc, float(kf["close"].iloc[-1]))["avg_cost"]
            except Exception:
                cost_lock = None

        # 关键指标行
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("现价", f"{r['现价']}")
        if cost_lock is not None:
            m2.metric(f"平均成本(锁仓{lock_strength:.1f})", f"{cost_lock:.2f}",
                      delta=f"经典 {r['平均成本']}", delta_color="off")
        else:
            m2.metric("平均成本", f"{r['平均成本']}", delta=f"溢价 {r['溢价%']}%")
        m3.metric("PE(TTM)", f"{r.get('pe_ttm', float('nan')):.1f}")
        m4.metric("综合分", f"{r['综合分']}")
        m5.metric("获利比例", f"{r['获利%']}%")
        m6.metric("集中度", f"{r['集中度']}")

        # 文字信息：股东户数近况 + 单季营收同比
        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("股东户数最近变化（户数降=筹码集中）")
            try:
                st.dataframe(load_holder_recent(code), use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"（户数明细获取失败：{type(e).__name__}）")
        with cc2:
            st.caption("过去连续四季 · 单季营收同比")
            try:
                st.dataframe(load_revenue(code), use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"（营收数据获取失败：{type(e).__name__}）")

        # 🏛️ 机构足迹：十大流通股东 + 北向持股（真实机构数据，对冲筹码模型的偏差）
        with st.expander("🏛️ 机构足迹（十大流通股东 + 北向持股）", expanded=False):
            gc1, gc2 = st.columns([3, 2])
            with gc1:
                try:
                    dt, top10 = load_top10(code)
                    if top10 is not None:
                        st.caption(f"十大流通股东（{dt}）· 『增减』>0=加仓，看『股东性质』里的基金/QFII/社保/机构")
                        st.dataframe(top10, use_container_width=True, hide_index=True)
                    else:
                        st.caption("（未取到十大流通股东数据）")
                except Exception as e:
                    st.caption(f"（十大股东获取失败：{type(e).__name__}）")
            with gc2:
                try:
                    nb, dcol, pcol = load_northbound(code)
                    if nb is not None and len(nb) and dcol and pcol:
                        nb2 = nb[[dcol, pcol]].dropna().copy()
                        nb2[dcol] = pd.to_datetime(nb2[dcol])
                        fign = go.Figure(go.Scatter(
                            x=nb2[dcol], y=pd.to_numeric(nb2[pcol], errors="coerce"),
                            mode="lines", line=dict(color="#e377c2")))
                        fign.update_layout(height=240, template="plotly_white",
                                           title="北向持股占A股%（至2024-08停更）",
                                           margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fign, use_container_width=True, key=f"nb_{key_prefix}_{code}")
                    else:
                        st.caption("（无北向持股数据）")
                except Exception as e:
                    st.caption(f"（北向数据获取失败：{type(e).__name__}）")
            st.caption("💡 真实机构足迹用来对冲筹码模型偏差：模型看不准庄家低位成本，"
                       "但十大股东的机构加仓 / 北向历史建仓能佐证『钱在哪、在哪个价位进的』。")


def _selected_from_plotly(ev, plot_df):
    try:
        pts = ev["selection"]["points"]
        if pts:
            pi = pts[0].get("point_index", pts[0].get("point_number"))
            if pi is not None and 0 <= pi < len(plot_df):
                return str(plot_df.iloc[pi]["代码"])
    except Exception:
        pass
    return None


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🧭 四象限互动图", "📊 结果表（筛选+多列排序）", "🔬 个股筹码详情",
     "🔥 各行业筹码冷热", "💰 成本低于现价分布"]
)

# ----------------------------- Tab1 四象限 ----------------------------- #
with tab1:
    plot_df = f.dropna(subset=["户数环比%", "溢价%"]).copy().reset_index(drop=True)
    if not len(plot_df):
        st.info("当前筛选无可绘制的数据。")
    else:
        plot_df["综合分_num"] = pd.to_numeric(plot_df["综合分"], errors="coerce")
        size = pd.to_numeric(plot_df["综合分"], errors="coerce").fillna(0) + 12
        fig = px.scatter(
            plot_df, x="户数环比%", y="溢价%",
            color="综合分_num", size=size, text="名称",
            color_continuous_scale="RdYlGn", range_color=[0, 100],
            hover_data={"代码": True, "综合分": True, "吸筹分": True, "筹码分": True,
                        "平均成本": True, "获利%": True, "集中度": True, "pe_ttm": True,
                        "象限": True, "综合分_num": False, "名称": False},
            labels={"户数环比%": "户数环比 %  (← 左=户数降/筹码集中)",
                    "溢价%": "溢价 %  (↑ 上=浮盈)", "综合分_num": "综合分"},
        )
        fig.update_traces(textposition="top center", textfont_size=9,
                          marker=dict(line=dict(width=0.8, color="rgba(0,0,0,0.5)")))
        fig.add_hline(y=p_thr, line_color="#888", line_width=1)
        fig.add_vline(x=h_thr, line_color="#888", line_width=1)
        # 横轴(户数环比%)对极端次新股做稳健裁剪：少数 +几万% 会把主群压成一条竖线
        xs = pd.to_numeric(plot_df["户数环比%"], errors="coerce")
        xlo = max(float(np.nanpercentile(xs, 1)), -60.0)
        xhi = min(float(np.nanpercentile(xs, 99)), 80.0)
        if not (np.isfinite(xlo) and np.isfinite(xhi)) or xlo >= xhi:
            xlo, xhi = -40.0, 40.0
        xpad = (xhi - xlo) * 0.06 + 1.0
        xr = [xlo - xpad, xhi + xpad]
        yr = [plot_df["溢价%"].min() - 3, plot_df["溢价%"].max() + 3]
        n_out = int((xs < xr[0]).sum() + (xs > xr[1]).sum())
        ann = [("强势控盘", xr[0], yr[1], "#2ca02c"), ("低位吸筹/筑底", xr[0], yr[0], "#1f77b4"),
               ("高位追高派发", xr[1], yr[1], "#ff7f0e"), ("套牢派发", xr[1], yr[0], "#d62728")]
        for txt, x, y, col in ann:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False,
                               font=dict(color=col, size=13), opacity=0.55,
                               xanchor="left" if x == xr[0] else "right")
        fig.update_xaxes(range=xr)
        fig.update_layout(height=640, template="plotly_white",
                          margin=dict(l=10, r=10, t=30, b=10), clickmode="event+select")
        ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="quad_sel")
        cap = "🖱️ 悬停看简要 · **点击圆球**在下方钉住该股 K线+详情 · 框选/滚轮缩放 · 双击复位"
        if n_out:
            cap += f"　|　已裁剪 {n_out} 只户数环比极端的次新股(超出视图,可在结果表查看)"
        st.caption(cap)
        picked = _selected_from_plotly(ev, plot_df)
        if picked:
            st.session_state["sel_code"] = picked
        st.divider()
        render_stock_preview(st.session_state.get("sel_code"), key_prefix="quad")

# ----------------------------- Tab2 结果表 ----------------------------- #
with tab2:
    show_cols = ["代码", "名称", "行业", "上市板", "象限", "综合分", "吸筹分", "筹码分", "户数环比%",
                 "现价", "溢价%", "获利%", "集中度", "集中度环比", "主峰",
                 "平均成本", "pe_ttm", "pb", "mktcap", "标签", "截止日"]
    show_cols = [c for c in show_cols if c in f.columns]
    NUM_COLS = [c for c in ["综合分", "吸筹分", "筹码分", "户数环比%", "现价", "溢价%",
                            "获利%", "集中度", "集中度环比", "主峰", "平均成本",
                            "pe_ttm", "pb", "mktcap"] if c in f.columns]

    preset_name = st.selectbox("⭐ 预设方案（一键套用筛选 + 排序）", list(PRESETS.keys()), index=0)
    preset = PRESETS[preset_name]

    if preset is None:
        # ---- 手动：TradingView 式任意列区间筛选（可多列叠加）----
        ff = f.copy()
        with st.expander("🎚️ 列区间筛选（可多列叠加，类似 TradingView 选股器）", expanded=True):
            pick = st.multiselect("选择要按区间筛选的列", NUM_COLS, default=["pe_ttm"])
            if pick:
                ui_cols = st.columns(min(3, len(pick)))
                for i, colname in enumerate(pick):
                    s = pd.to_numeric(f[colname], errors="coerce")
                    lo, hi = (float(np.nanmin(s)), float(np.nanmax(s))) if s.notna().any() else (0.0, 0.0)
                    with ui_cols[i % len(ui_cols)]:
                        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                            rng = st.slider(colname, lo, hi, (lo, hi), key=f"flt_{colname}")
                        else:
                            rng = (lo, hi)
                        keep_nan = st.checkbox(f"含{colname}缺失", value=True, key=f"nan_{colname}")
                    sv = pd.to_numeric(ff[colname], errors="coerce")
                    mask = sv.between(rng[0], rng[1])
                    if keep_nan:
                        mask = mask | sv.isna()
                    ff = ff[mask]
        sc1, sc2 = st.columns(2)
        sortable = [c for c in show_cols if c not in ("代码", "名称", "标签", "截止日")]
        sort_cols = sc1.multiselect("排序列（按选择顺序为优先级，可多列）", sortable, default=["综合分"])
        asc_cols = sc2.multiselect("其中用『升序』的列（其余降序）", sort_cols)
    else:
        # ---- 预设：从『行业+上市板』全集出发（忽略侧栏象限），按预设条件直接套用 ----
        ff = f_ind.copy()
        desc = []
        if preset.get("quad"):
            ff = ff[ff["象限"].isin(preset["quad"])]
            desc.append("象限=" + "/".join(preset["quad"]))
        for coln, (lo, hi) in preset.get("num", {}).items():
            if coln in ff.columns:
                sv = pd.to_numeric(ff[coln], errors="coerce")
                ff = ff[sv.between(lo, hi)]   # 预设数值条件默认剔除缺失
                hir = "∞" if hi >= 1e9 else hi
                desc.append(f"{coln}∈[{lo},{hir}]")
        sort_cols = [c for c in preset.get("sort", ["综合分"]) if c in show_cols]
        asc_cols = sort_cols if preset.get("asc") else []
        st.success(f"已套用预设：**{preset_name}**　条件：{'；'.join(desc)}"
                   "　（忽略侧栏『象限』；行业/上市板筛选仍生效）")

    tbl = ff[show_cols].copy()
    if sort_cols:
        ascending = [c in asc_cols for c in sort_cols]
        tbl = tbl.sort_values(by=sort_cols, ascending=ascending, na_position="last")
    tbl = tbl.reset_index(drop=True)
    prio = " → ".join(f"{c}{'↑' if c in asc_cols else '↓'}" for c in sort_cols) or "（无）"
    st.caption(f"筛选后 {len(tbl)} 只 · 排序优先级：{prio}")
    tbl_ev = st.dataframe(
        tbl, use_container_width=True, height=480, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="tbl_sel",
        column_config={
            "综合分": st.column_config.ProgressColumn("综合分", min_value=0, max_value=100, format="%d"),
            "pe_ttm": st.column_config.NumberColumn("PE(TTM)", format="%.1f"),
            "mktcap": st.column_config.NumberColumn("总市值(亿)", format="%.0f"),
            "pb": st.column_config.NumberColumn("PB", format="%.2f"),
        },
    )
    try:
        _rows = tbl_ev["selection"]["rows"]
        if _rows:
            st.session_state["sel_code"] = str(tbl.iloc[_rows[0]]["代码"])
    except Exception:
        pass
    st.caption("👆 **点某一行**在下方钉住该股 K线+详情；点列名排序；右上角可下载。")
    st.download_button("⬇️ 导出当前结果 CSV", tbl.to_csv(index=False).encode("utf-8-sig"),
                       file_name="techscan_result.csv", mime="text/csv")
    st.divider()
    render_stock_preview(st.session_state.get("sel_code"), key_prefix="tbl")

# ----------------------------- Tab3 个股筹码详情 ----------------------------- #
with tab3:
    if not len(prices):
        st.info("无价格缓存，无法画个股筹码图。")
    else:
        names = (f["代码"] + " " + f["名称"]).tolist() or (view["代码"] + " " + view["名称"]).tolist()
        pick = st.selectbox("选择个股", names)
        code = pick.split()[0]
        pdf = prices[prices["代码"] == code].sort_values("date")
        if not len(pdf):
            st.warning("该票无价格数据。")
        else:
            cur = float(pdf["close"].iloc[-1])
            g, c = compute_chip_distribution(pdf, decay_coef=float(decay), lock_strength=lock_strength)
            m = chip_metrics(g, c, cur)
            colA, colB = st.columns([2, 1])
            with colA:
                col_bar = np.where(g <= cur, "#d62728", "#2ca02c")
                fig2 = go.Figure(go.Bar(x=c, y=g, orientation="h",
                                        marker_color=col_bar, hovertemplate="价位%{y:.2f}<br>占比%{x:.4f}<extra></extra>"))
                fig2.add_hline(y=cur, line_dash="dash", line_color="#333",
                               annotation_text=f"现价 {cur:.2f}")
                fig2.add_hline(y=m["avg_cost"], line_dash="dot", line_color="#9467bd",
                               annotation_text=f"平均成本 {m['avg_cost']:.2f}")
                fig2.add_hrect(y0=m["cost_low"], y1=m["cost_high"], fillcolor="#1f77b4",
                               opacity=0.08, line_width=0)
                fig2.update_layout(height=600, template="plotly_white",
                                   title=f"{pick} 筹码分布 (decay={decay} · 锁仓{lock_strength:.1f})",
                                   xaxis_title="筹码占比", yaxis_title="价格",
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig2, use_container_width=True)
            with colB:
                st.metric("现价", f"{cur:.2f}")
                st.metric("平均成本", f"{m['avg_cost']:.2f}",
                          delta=f"溢价 {(cur/m['avg_cost']-1)*100:+.1f}%")
                st.metric("获利比例", f"{m['profit_ratio']*100:.1f}%")
                st.metric("筹码主峰", f"{m['peak_price']:.2f}")
                st.metric("集中度", f"{m['concentration']:.3f}")
                st.metric("90%成本区间", f"{m['cost_low']:.2f} ~ {m['cost_high']:.2f}")
                rowinfo = view[view["代码"] == code]
                if len(rowinfo):
                    r0 = rowinfo.iloc[0]
                    st.metric("行业", f"{r0.get('行业', '—')}")
                    st.metric("户数环比", f"{r0['户数环比%']}%")
                    st.metric("吸筹分 / 综合分", f"{r0['吸筹分']} / {r0['综合分']}")
                    st.metric("PE(TTM) / PB", f"{r0.get('pe_ttm', np.nan):.1f} / {r0.get('pb', np.nan):.2f}")

# ----------------------------- Tab4 各行业筹码冷热 ----------------------------- #
with tab4:
    st.caption("📌 本页覆盖当前『行业 + 上市板』筛选下的全部股票，**不受『象限』筛选影响**。")
    if not len(f_ind):
        st.info("当前筛选无数据。")
    else:
        dim = st.radio("分组维度", ["行业", "上市板"], horizontal=True,
                       help="可切换按申万行业 或 按上市板(主板/创业板/科创板)对比")
        g = f_ind.copy()
        for col in ["综合分", "吸筹分", "筹码分", "获利%", "集中度", "溢价%", "户数环比%", "pe_ttm"]:
            g[col] = pd.to_numeric(g[col], errors="coerce")
        agg = g.groupby(dim).agg(
            数量=("代码", "count"),
            综合分_均=("综合分", "mean"),
            获利_均=("获利%", "mean"),
            集中度_均=("集中度", "mean"),
            溢价_均=("溢价%", "mean"),
            户数环比_均=("户数环比%", "mean"),
            PE中位=("pe_ttm", "median"),
        ).round(2).reset_index().sort_values("综合分_均", ascending=False)

        ca, cb = st.columns([1, 1])
        with ca:
            st.markdown(f"**各{dim}冷热汇总**")
            st.dataframe(agg, use_container_width=True, hide_index=True)
        with cb:
            quad_ct = g.pivot_table(index=dim, columns="象限", values="代码",
                                    aggfunc="count", fill_value=0)
            order = [q for q in ["强势控盘", "低位吸筹/筑底", "高位追高派发", "套牢派发"] if q in quad_ct.columns]
            quad_ct = quad_ct[order] if order else quad_ct
            figb = go.Figure()
            for q in quad_ct.columns:
                figb.add_bar(x=quad_ct.index, y=quad_ct[q], name=q,
                             marker_color=QUAD_COLORS.get(q))
            figb.update_layout(barmode="stack", height=360, template="plotly_white",
                               title=f"各{dim} · 象限分布（堆叠）",
                               margin=dict(l=10, r=10, t=40, b=10), legend_title="")
            st.plotly_chart(figb, use_container_width=True)

        # 上市板维度时，额外给一组关键指标对比柱（科创 vs 创业 vs 主板）
        if dim == "上市板":
            mcmp = agg.melt(id_vars="上市板",
                            value_vars=["综合分_均", "获利_均", "溢价_均", "户数环比_均"],
                            var_name="指标", value_name="值")
            figm = px.bar(mcmp, x="指标", y="值", color="上市板", barmode="group",
                          title="上市板关键指标对比（均值）")
            figm.update_layout(height=360, template="plotly_white",
                               margin=dict(l=10, r=10, t=40, b=10), legend_title="")
            st.plotly_chart(figm, use_container_width=True)

        sub = g.dropna(subset=["户数环比%", "溢价%"])
        fig_heat = px.scatter(
            sub, x="户数环比%", y="溢价%",
            color=dim, size=pd.to_numeric(sub["综合分"], errors="coerce").fillna(0) + 10,
            hover_data=["代码", "名称", "行业", "上市板", "综合分"], opacity=0.75,
            labels={"户数环比%": "户数环比 %  (←左=户数降)", "溢价%": "溢价 %  (↑上=浮盈)"},
        )
        fig_heat.add_hline(y=p_thr, line_color="#888"); fig_heat.add_vline(x=h_thr, line_color="#888")
        _xs = pd.to_numeric(sub["户数环比%"], errors="coerce")
        _xlo, _xhi = max(float(np.nanpercentile(_xs, 1)), -60.0), min(float(np.nanpercentile(_xs, 99)), 80.0)
        if np.isfinite(_xlo) and np.isfinite(_xhi) and _xlo < _xhi:
            fig_heat.update_xaxes(range=[_xlo - 3, _xhi + 3])
        fig_heat.update_layout(height=520, template="plotly_white",
                               title=f"按{dim}着色的四象限分布", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_heat, use_container_width=True)

# ----------------------------- Tab5 成本低于现价分布 ----------------------------- #
with tab5:
    st.markdown("#### 平均成本低于现价的幅度分布（浮盈安全垫）")
    st.caption("低于现价% = (现价 − 平均成本) / 现价 × 100；>0 表示平均持仓成本在现价之下（浮盈垫），"
               "数值越大安全垫越厚。平均成本随左侧『衰减系数』变化。")
    st.caption("📌 本页覆盖当前『行业』筛选下的全部股票，**不受『象限』筛选影响**。")
    g = f_ind.copy()
    g["现价n"] = pd.to_numeric(g["现价"], errors="coerce")
    g["成本n"] = pd.to_numeric(g["平均成本"], errors="coerce")
    g = g[(g["现价n"] > 0) & g["成本n"].notna()]
    g["低于现价%"] = ((g["现价n"] - g["成本n"]) / g["现价n"] * 100).round(1)

    prof = g[g["低于现价%"] > 0].copy()
    trapped = g[g["低于现价%"] <= 0].copy()
    width = st.select_slider("区间宽度 %（分桶粒度）", options=[5, 10, 15, 20, 25], value=10)
    maxv = float(prof["低于现价%"].max()) if len(prof) else width
    ncut = min(max(1, int(np.ceil(maxv / width))), 20)
    edges = [i * width for i in range(ncut + 1)]
    BINS = edges + [float("inf")]
    LABELS = [f"{edges[i]}-{edges[i + 1]}%" for i in range(ncut)] + [f"{edges[-1]}%+"]
    prof["区间"] = pd.cut(prof["低于现价%"], bins=BINS, labels=LABELS, right=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("浮盈股票数（成本<现价）", f"{len(prof)} 只")
    k2.metric("套牢股票数（成本≥现价）", f"{len(trapped)} 只")
    k3.metric("浮盈占比", f"{(len(prof)/max(len(g),1)*100):.0f}%")
    k4.metric("低于现价% 中位", f"{g['低于现价%'].median():.1f}%")

    cnt = prof.groupby("区间", observed=False)["代码"].count().reindex(LABELS).fillna(0).astype(int)
    colL, colR = st.columns([1, 1])
    with colL:
        figc = go.Figure(go.Bar(
            x=LABELS, y=cnt.values, text=cnt.values, textposition="outside",
            marker=dict(color=list(range(len(LABELS))), colorscale="PuRd")))
        figc.update_layout(height=380, template="plotly_white",
                           title=f"各『低于现价%』区间的股票数量（每档 {width}%）",
                           xaxis_title="平均成本低于现价的幅度", yaxis_title="股票数",
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figc, use_container_width=True)
    with colR:
        seg = prof.pivot_table(index="区间", columns="行业", values="代码",
                               aggfunc="count", fill_value=0, observed=False).reindex(LABELS).fillna(0)
        figs = go.Figure()
        for ind in seg.columns:
            figs.add_bar(x=LABELS, y=seg[ind].values, name=ind)
        figs.update_layout(barmode="stack", height=380, template="plotly_white",
                           title="各区间 · 行业构成（堆叠）",
                           margin=dict(l=10, r=10, t=40, b=10), legend_title="")
        st.plotly_chart(figs, use_container_width=True)

    st.markdown("**浮盈股票明细**（按低于现价% 降序，可点列名排序）")
    cols5 = [c for c in ["代码", "名称", "行业", "区间", "低于现价%", "现价", "平均成本",
                         "溢价%", "综合分", "获利%", "集中度", "pe_ttm"] if c in prof.columns]
    detail = prof[cols5].sort_values("低于现价%", ascending=False).reset_index(drop=True)
    st.dataframe(detail, use_container_width=True, height=420, hide_index=True,
                 column_config={"pe_ttm": st.column_config.NumberColumn("PE(TTM)", format="%.1f")})
    st.download_button("⬇️ 导出浮盈明细 CSV", detail.to_csv(index=False).encode("utf-8-sig"),
                       file_name="cost_below_price.csv", mime="text/csv")
    with st.expander(f"套牢股票（平均成本≥现价）· {len(trapped)} 只"):
        tcols = [c for c in ["代码", "名称", "行业", "低于现价%", "现价", "平均成本", "溢价%", "综合分"] if c in trapped.columns]
        st.dataframe(trapped[tcols].sort_values("低于现价%").reset_index(drop=True),
                     use_container_width=True, hide_index=True)
