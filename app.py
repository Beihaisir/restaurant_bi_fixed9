from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.io import read_any_table, read_raw_noheader
from utils.rules import load_rules_xlsx, match_categories, Rule
from utils.transform import build_fact_tables

st.set_page_config(page_title="餐饮经营分析系统", layout="wide")


@dataclass
class StoreBundle:
    store_id: str
    daily: Optional[pd.DataFrame]
    dish: Optional[pd.DataFrame]
    pay: Optional[pd.DataFrame]


DAILY_MUST = ["门店代码", "门店名称", "日期"]
DISH_MUST = ["创建时间", "菜品名称", "POS销售单号"]
PAY_MUST = ["POS销售单号", "支付类型", "总金额"]


def _as_bytes(uploaded) -> bytes:
    return uploaded.getvalue()


def _norm_colset(cols) -> set:
    out = set()
    for c in cols:
        s = str(c).strip().replace(" ", "").replace("\u3000", "")
        out.add(s)
    return out


def detect_table_kind(file_bytes: bytes, filename: str) -> Tuple[str, Optional[str]]:
    """
    Returns (kind, store_id) where kind in {"daily","dish","pay","unknown"}.
    Robust heuristics for real-world exported Excel.
    """
    raw = read_raw_noheader(file_bytes, filename)

    # store id
    store_id = None
    for r in range(min(20, len(raw))):
        for cell in raw.iloc[r].astype(str).tolist():
            if "导出人" in str(cell):
                import re

                m = re.search(r"导出人[:：]\s*(\d+)", str(cell))
                if m:
                    store_id = m.group(1)
                    break
        if store_id:
            break

    top_text = " ".join(raw.head(5).astype(str).fillna("").values.flatten().tolist())
    if "日销售报表" in top_text:
        return "daily", store_id

    df_dish, _ = read_any_table(file_bytes, filename, DISH_MUST)
    if {"POS销售单号", "菜品名称", "创建时间"}.issubset(_norm_colset(df_dish.columns)):
        return "dish", store_id

    df_pay, _ = read_any_table(file_bytes, filename, PAY_MUST)
    if {"POS销售单号", "支付类型"}.issubset(_norm_colset(df_pay.columns)):
        return "pay", store_id

    df_daily, _ = read_any_table(file_bytes, filename, DAILY_MUST)
    cols_daily = _norm_colset(df_daily.columns)
    if ("含税销售额" in cols_daily) or ("客流量" in cols_daily) or ({"门店代码", "门店名称", "日期"}.issubset(cols_daily)):
        return "daily", store_id

    return "unknown", store_id


@st.cache_data(show_spinner=False)
def parse_uploaded(files: List, rule_file) -> Tuple[List[StoreBundle], List[str], List[Rule]]:
    rules: List[Rule] = []
    if rule_file is not None:
        rules = load_rules_xlsx(io.BytesIO(_as_bytes(rule_file)))

    bundles: Dict[str, StoreBundle] = {}
    warnings: List[str] = []

    def upsert(store_id: str) -> StoreBundle:
        if store_id not in bundles:
            bundles[store_id] = StoreBundle(store_id=store_id, daily=None, dish=None, pay=None)
        return bundles[store_id]

    for f in files:
        b = _as_bytes(f)
        name = f.name
        kind, store_id = detect_table_kind(b, name)
        if store_id is None:
            store_id = "UNKNOWN"

        if kind == "daily":
            df, _ = read_any_table(b, name, DAILY_MUST)
            upsert(store_id).daily = df
        elif kind == "dish":
            df, _ = read_any_table(b, name, DISH_MUST)
            upsert(store_id).dish = df
        elif kind == "pay":
            df, _ = read_any_table(b, name, PAY_MUST)
            upsert(store_id).pay = df
        else:
            warnings.append(f"无法识别文件类型：{name}（已跳过）")

    out = list(bundles.values())
    out.sort(key=lambda x: x.store_id)
    return out, warnings, rules


def fmt_money(x: float) -> str:
    try:
        return f"¥{x:,.2f}"
    except Exception:
        return "—"


def halfhour_options(min_dt: pd.Timestamp, max_dt: pd.Timestamp) -> List[pd.Timestamp]:
    if pd.isna(min_dt) or pd.isna(max_dt):
        return []
    start = min_dt.floor("30min")
    end = max_dt.ceil("30min")
    return list(pd.date_range(start, end, freq="30min"))


def apply_time_filter(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    x = df.copy()
    if col not in x.columns:
        return x
    x[col] = pd.to_datetime(x[col], errors="coerce")
    return x[(x[col] >= start) & (x[col] <= end)].copy()


def _base_items(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if "类型_norm" in df.columns:
        return df[df["类型_norm"].isin(["菜品", "套餐"])].copy()
    # fallback: contains
    return df[df["类型"].astype(str).str.contains("菜品|套餐", na=False)].copy()


def main() -> None:
    st.title("🍽️ 餐饮经营分析系统（连锁视角 · 董事/股东 · 门店店长）")

    with st.sidebar:
        st.header("数据输入")
        rule_file = st.file_uploader("上传：分类规则模板（xlsx，Sheet=规则表）", type=["xlsx"], accept_multiple_files=False)
        files = st.file_uploader(
            "上传：三类报表（可多门店、多文件；支持 xls/xlsx/csv）",
            type=["xls", "xlsx", "csv"],
            accept_multiple_files=True,
        )
        st.caption("口径：时间最小30分钟；“加xx”为单加（加多宝除外）；天麻面并入细面；“标准”仅统计为【套餐】的标准行。")

    if not files:
        st.info("请先在左侧上传报表文件。")
        return

    bundles, warnings, rules = parse_uploaded(files, rule_file)

    if warnings:
        with st.expander("⚠️ 文件识别警告", expanded=False):
            for w in warnings:
                st.warning(w)

    analyzable = [b for b in bundles if b.dish is not None and b.pay is not None and b.daily is not None and b.store_id != "UNKNOWN"]
    missing = [b for b in bundles if b not in analyzable]

    if missing:
        with st.expander("⚠️ 缺表门店（不进入分析）", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "store_id": b.store_id,
                            "有日销售": b.daily is not None,
                            "有菜品明细": b.dish is not None,
                            "有支付明细": b.pay is not None,
                        }
                        for b in missing
                    ]
                ),
                use_container_width=True,
            )

    if not analyzable:
        st.error("没有“三表齐全”的门店，无法分析。")
        return

    store_ids = [b.store_id for b in analyzable]
    sel_stores = st.multiselect("选择门店（支持多店对比）", options=store_ids, default=store_ids[:1])
    if not sel_stores:
        st.stop()

    facts_by_store: Dict[str, Dict[str, pd.DataFrame]] = {}
    for b in analyzable:
        if b.store_id in sel_stores:
            facts_by_store[b.store_id] = build_fact_tables(b.dish, b.pay, rules, b.store_id)

    all_orders = pd.concat([facts_by_store[s]["fact_orders"] for s in sel_stores], ignore_index=True)
    min_dt = all_orders["order_time"].min()
    max_dt = all_orders["order_time"].max()
    opts = halfhour_options(min_dt, max_dt)
    if not opts:
        st.error("无法从数据中解析创建时间。")
        return

    c1, c2 = st.columns(2)
    with c1:
        start = st.selectbox("开始时间（30分钟粒度）", options=opts, index=0, format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))
    with c2:
        end = st.selectbox("结束时间（30分钟粒度）", options=opts, index=len(opts) - 1, format_func=lambda x: x.strftime("%Y-%m-%d %H:%M"))

    if start > end:
        st.error("开始时间不能晚于结束时间。")
        return

    filtered: Dict[str, Dict[str, pd.DataFrame]] = {}
    for sid in sel_stores:
        f = facts_by_store[sid]
        filtered[sid] = {
            "items_main": apply_time_filter(f["fact_items_main"], "创建时间", start, end),
            "items_add": apply_time_filter(f["fact_items_add"], "created_at", start, end),
            "pay": apply_time_filter(f["fact_pay"], "order_time", start, end),
            "orders": apply_time_filter(f["fact_orders"], "order_time", start, end),
        }

    tabs = st.tabs(
        [
            "① 董事/股东总览",
            "② 门店对比",
            "③ 规格",
            "④ 品类结构",
            "⑤ 单加分析",
            "⑥ 支付渠道",
            "⑦ 退款/异常与对账",
            "⑧ 未分类池（可导出）",
            "⑨ 明细导出",
            "⑩ 时段热力图",
        ]
    )

    # ① 总览（规模、效率、爆品/长尾、单加结构）
    with tabs[0]:
        st.subheader("董事/股东视角：规模、效率、结构、风险")

        rows = []
        for sid in sel_stores:
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            orders = int(o["POS销售单号"].nunique()) if not o.empty else 0
            rows.append(
                {
                    "store_id": sid,
                    "订单数": orders,
                    "菜品销量": float(o["dish_qty"].sum()) if not o.empty else 0.0,
                    "菜品应收(优惠后)": float(o["net_amount"].sum()) if not o.empty else 0.0,
                    "支付实收": float(p["总金额"].sum()) if not p.empty else 0.0,
                    "退款单占比": float(o["has_refund"].mean()) if not o.empty else 0.0,
                    "客单(应收/订单)": (float(o["net_amount"].sum()) / orders) if orders else np.nan,
                }
            )
        dfk = pd.DataFrame(rows)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("选中门店订单数", int(dfk["订单数"].sum()))
        k2.metric("选中门店菜品销量", f"{dfk['菜品销量'].sum():,.0f}")
        k3.metric("选中门店菜品应收(优惠后)", fmt_money(dfk["菜品应收(优惠后)"].sum()))
        k4.metric("选中门店支付实收", fmt_money(dfk["支付实收"].sum()))
        st.dataframe(dfk, use_container_width=True)

        # 半小时趋势 + 峰值
        oall = pd.concat([filtered[s]["orders"] for s in sel_stores], ignore_index=True)
        if not oall.empty:
            oall["bucket"] = oall["order_time"].dt.floor("30min")
            trend = oall.groupby("bucket", as_index=False).agg(
                订单数=("POS销售单号", "nunique"),
                菜品应收=("net_amount", "sum"),
            ).sort_values("bucket")
            st.line_chart(trend.set_index("bucket")[["订单数", "菜品应收"]])
            peak = trend.sort_values("订单数", ascending=False).head(10)
            st.markdown("**峰值时段 Top10（按订单数）**")
            st.dataframe(peak, use_container_width=True)

        # Top 菜品 + 帕累托
        main_all = pd.concat([filtered[s]["items_main"] for s in sel_stores], ignore_index=True)
        base_items = _base_items(main_all)
        if not base_items.empty:
            top_rev = (
                base_items.groupby("菜品名称", as_index=False)
                .agg(应收=("优惠后小计价格", "sum"), 销量=("菜品数量", "sum"), 订单数=("POS销售单号", "nunique"))
                .sort_values(["应收", "销量"], ascending=False)
                .head(20)
            )
            st.markdown("### Top20 菜品（按应收排序）")
            st.dataframe(top_rev, use_container_width=True)
            st.bar_chart(top_rev.set_index("菜品名称")[["应收"]])

            dish_rev = base_items.groupby("菜品名称", as_index=False).agg(应收=("优惠后小计价格", "sum")).sort_values("应收", ascending=False)
            dish_rev["累计应收"] = dish_rev["应收"].cumsum()
            total_rev = dish_rev["应收"].sum()
            dish_rev["累计占比"] = dish_rev["累计应收"] / total_rev if total_rev else 0
            n80 = int((dish_rev["累计占比"] <= 0.8).sum() + 1) if total_rev else 0
            st.markdown("### 爆品/长尾（帕累托）")
            st.write(f"达到 **80%应收** 需要的菜品数：**{n80}** / 总菜品数 {len(dish_rev)}")
            st.dataframe(dish_rev.head(50), use_container_width=True)

        # Top 单加
        add_all = pd.concat([filtered[s]["items_add"] for s in sel_stores], ignore_index=True)
        if not add_all.empty:
            top_add = (
                add_all.groupby("add_display", as_index=False)
                .agg(单加金额=("amount", "sum"), 销量=("qty", "sum"), 订单数=("order_id", "nunique"))
                .sort_values(["单加金额", "销量"], ascending=False)
                .head(20)
            )
            st.markdown("### Top20 单加（按单加金额排序）")
            st.dataframe(top_add, use_container_width=True)
            st.bar_chart(top_add.set_index("add_display")[["单加金额"]])

    # ② 门店对比
    with tabs[1]:
        st.subheader("门店对比：同口径看差异（店长/区域经理/总部）")
        rows = []
        for sid in sel_stores:
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            orders = int(o["POS销售单号"].nunique()) if not o.empty else 0
            net = float(o["net_amount"].sum()) if not o.empty else 0.0
            paid = float(p["总金额"].sum()) if not p.empty else 0.0
            rows.append(
                {
                    "store_id": sid,
                    "订单数": orders,
                    "应收(优惠后)": net,
                    "实收": paid,
                    "应收-实收差异": net - paid,
                    "客单(应收/订单)": (net / orders) if orders else np.nan,
                }
            )
        df = pd.DataFrame(rows).sort_values("应收(优惠后)", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("store_id")[["应收(优惠后)", "实收"]])
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(df.set_index("store_id")[["应收-实收差异"]])
        with c2:
            st.bar_chart(df.set_index("store_id")[["客单(应收/订单)"]])

    # ③ 规格（含标准=套餐标准）
    with tabs[2]:
        st.subheader("规格：主食结构（含“标准”=套餐标准）")
        st.caption("规格分布只统计：标准 / 宽面 / 细面(含天麻面) / 米饭 / 宽粉(含粉) / 无需主食；“标准”仅来源于 类型=套餐 的标准行。")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue

            base = _base_items(m)
            spec_base = base[base["spec_norm"].notna()].copy()
            if spec_base.empty:
                st.info("该时间范围内没有命中规格白名单的数据。")
                continue

            spec = (
                spec_base.groupby("spec_norm", as_index=False)
                .agg(销量=("菜品数量", "sum"), 应收=("优惠后小计价格", "sum"), 行数=("菜品名称", "count"), 订单数=("POS销售单号", "nunique"))
                .sort_values(["销量", "应收"], ascending=False)
            )
            spec["销量占比"] = spec["销量"] / spec["销量"].sum() if spec["销量"].sum() else 0
            spec["应收占比"] = spec["应收"] / spec["应收"].sum() if spec["应收"].sum() else 0
            st.dataframe(spec, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.bar_chart(spec.set_index("spec_norm")[["销量"]])
            with c2:
                st.bar_chart(spec.set_index("spec_norm")[["应收"]])

            spec_base["bucket"] = spec_base["创建时间"].dt.floor("30min")
            top_specs = spec["spec_norm"].head(5).tolist()
            pivot = (
                spec_base[spec_base["spec_norm"].isin(top_specs)]
                .groupby(["bucket", "spec_norm"], as_index=False)
                .agg(销量=("菜品数量", "sum"))
            )
            if not pivot.empty:
                piv = pivot.pivot(index="bucket", columns="spec_norm", values="销量").fillna(0).sort_index()
                st.line_chart(piv)

    # ④ 品类结构
    with tabs[3]:
        st.subheader("品类结构：规则模板命中（多标签）")
        st.caption("一个菜品可命中多个分类，命中即各计一次；未命中进入未分类池。")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue

            exploded = m.copy().explode("categories")
            exploded["categories"] = exploded["categories"].fillna("未分类")

            cat = (
                exploded.groupby("categories", as_index=False)
                .agg(销量=("菜品数量", "sum"), 应收=("优惠后小计价格", "sum"), 菜品行数=("菜品名称", "count"))
                .sort_values("应收", ascending=False)
            )

            st.dataframe(cat, use_container_width=True)
            st.bar_chart(cat.set_index("categories")[["应收"]])

            topn = st.slider(f"TopN 菜品（门店 {sid}）", min_value=5, max_value=50, value=20, step=5, key=f"topn_{sid}")
            cats = ["全部"] + sorted(exploded["categories"].dropna().unique().tolist())
            sel_cat = st.selectbox(f"选择分类（门店 {sid}）", options=cats, key=f"selcat_{sid}")
            view = exploded if sel_cat == "全部" else exploded[exploded["categories"] == sel_cat]
            top_items = (
                view.groupby("菜品名称", as_index=False)
                .agg(应收=("优惠后小计价格", "sum"), 销量=("菜品数量", "sum"), 订单数=("POS销售单号", "nunique"))
                .sort_values(["应收", "销量"], ascending=False)
                .head(topn)
            )
            st.dataframe(top_items, use_container_width=True)

    # ⑤ 单加分析
    with tabs[4]:
        st.subheader("单加分析：加料带来的结构与客单提升（与主菜严格隔离）")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            a = filtered[sid]["items_add"]
            if a.empty:
                st.info("无单加记录")
                continue

            add = (
                a.groupby("add_display", as_index=False)
                .agg(销量=("qty", "sum"), 单加金额=("amount", "sum"), 订单数=("order_id", "nunique"), 来源=("source", lambda s: ",".join(sorted(set(map(str, s))))))
                .sort_values(["单加金额", "销量"], ascending=False)
            )
            st.dataframe(add, use_container_width=True)
            st.bar_chart(add.set_index("add_display")[["单加金额"]])

            orders = filtered[sid]["orders"]
            add_orders = int(a["order_id"].nunique())
            total_orders = int(orders["POS销售单号"].nunique()) if not orders.empty else 0
            st.metric("单加渗透率（含单加订单/总订单）", f"{(add_orders / total_orders * 100) if total_orders else 0:.1f}%")

            # 单加对客单提升（订单层面）
            if not orders.empty:
                add_set = set(a["order_id"].dropna().astype(str).tolist())
                o2 = orders.copy()
                o2["has_add"] = o2["POS销售单号"].astype(str).isin(add_set)
                grp = o2.groupby("has_add", as_index=False).agg(订单数=("POS销售单号", "nunique"), 应收=("net_amount", "sum"))
                grp["客单(应收/订单)"] = grp["应收"] / grp["订单数"].replace(0, np.nan)
                st.markdown("**有单加 vs 无单加（客单提升）**")
                st.dataframe(grp, use_container_width=True)

    # ⑥ 支付渠道
    with tabs[5]:
        st.subheader("支付渠道：渠道结构、团购渗透、混合支付")
        st.caption("排序按实收降序，其次按支付笔数。若实收=0，会展示支付明细前50行便于排查（金额列名/格式/时间范围）。")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            p = filtered[sid]["pay"]
            if p.empty:
                st.warning("无支付数据（该门店在筛选时间范围内支付表未关联到任何订单，或支付表未被正确识别）")
                continue

            if float(p["总金额"].sum()) == 0.0:
                st.warning("已读到支付表，但总金额解析为 0：可能是金额列名/格式不一致，或该时间范围内无匹配订单。")
                st.dataframe(p.head(50), use_container_width=True)

            pay = (
                p.groupby("支付类型", as_index=False)
                .agg(实收=("总金额", "sum"), 支付笔数=("POS销售单号", "count"), 涉及订单=("POS销售单号", "nunique"))
                .sort_values(["实收", "支付笔数"], ascending=False)
            )
            st.dataframe(pay, use_container_width=True)
            st.bar_chart(pay.set_index("支付类型")[["实收", "支付笔数"]])

            multi = p.groupby("POS销售单号").agg(n=("支付类型", "nunique")).reset_index()
            mixed_rate = (multi["n"] > 1).mean() if not multi.empty else 0
            st.metric("混合支付订单占比", f"{mixed_rate * 100:.1f}%")

    # ⑦ 退款/异常与对账（差异分解）
    with tabs[6]:
        st.subheader("退款/异常与对账：抓风险、抓漏损、抓口径问题")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            o = filtered[sid]["orders"]
            p = filtered[sid]["pay"]
            if o.empty:
                st.info("无数据")
                continue

            r = o.merge(p.groupby(["store_id", "POS销售单号"], as_index=False).agg(paid=("总金额", "sum")), on=["store_id", "POS销售单号"], how="left")
            r["paid"] = r["paid"].fillna(0.0)
            r["diff"] = r["net_amount"] - r["paid"]

            st.caption("对账口径：菜品应收（优惠后小计求和） vs 支付实收（支付表金额求和）。")
            st.dataframe(r.sort_values("diff", ascending=False).head(100), use_container_width=True)

            # 差异分解：按渠道/按半小时
            if not p.empty:
                pay_kind = p.groupby(["store_id", "POS销售单号"], as_index=False).agg(
                    paid=("总金额", "sum"),
                    k=("支付类型", lambda s: "混合" if s.nunique() > 1 else str(list(s)[0])),
                )
                rr = o.merge(pay_kind, on=["store_id", "POS销售单号"], how="left")
                rr["paid"] = rr["paid"].fillna(0.0)
                rr["diff"] = rr["net_amount"] - rr["paid"]

                byk = (
                    rr.groupby("k", as_index=False)
                    .agg(订单数=("POS销售单号", "nunique"), 应收=("net_amount", "sum"), 实收=("paid", "sum"), 差异=("diff", "sum"))
                    .sort_values("差异", ascending=False)
                )
                st.markdown("**按支付渠道分解（差异=应收-实收）**")
                st.dataframe(byk, use_container_width=True)
                st.bar_chart(byk.set_index("k")[["差异"]])

                rr["slot"] = rr["order_time"].dt.floor("30min")
                bys = (
                    rr.groupby("slot", as_index=False)
                    .agg(订单数=("POS销售单号", "nunique"), 差异=("diff", "sum"), 应收=("net_amount", "sum"), 实收=("paid", "sum"))
                    .sort_values("slot")
                )
                st.markdown("**按半小时分解（差异趋势）**")
                st.line_chart(bys.set_index("slot")[["差异", "应收", "实收"]])

                st.download_button(
                    f"导出差异Top订单（{sid}）CSV",
                    data=rr.sort_values("diff", ascending=False).head(500).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"对账差异Top_{sid}.csv",
                    mime="text/csv",
                )

            refund_rate = float(o["has_refund"].mean()) if not o.empty else 0.0
            st.metric("退款单占比（菜品表存在POS退款单号）", f"{refund_rate * 100:.1f}%")

    # ⑧ 未分类池
    with tabs[7]:
        st.subheader("未分类池：可查看、可导出（规则迭代入口）")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            if m.empty:
                st.info("无数据")
                continue

            un = m[m["categories"].apply(lambda x: len(x) == 0)].copy()
            st.write(f"未分类主菜行数：{len(un):,}")
            st.dataframe(un[["创建时间", "菜品名称", "类型", "规格名称", "优惠后小计价格", "POS销售单号"]].head(200), use_container_width=True)
            st.download_button(
                f"导出未分类主菜（{sid}）CSV",
                data=un.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"未分类主菜_{sid}.csv",
                mime="text/csv",
            )

            a = filtered[sid]["items_add"]
            if not a.empty:
                if rules:
                    a2 = a.copy()
                    a2["add_cat"] = a2["add_name"].apply(lambda x: " / ".join(match_categories(str(x), rules)) if match_categories(str(x), rules) else "")
                    un_add = a2[a2["add_cat"] == ""].copy()
                else:
                    un_add = a.copy()

                st.write(f"未分类单加行数：{len(un_add):,}")
                st.dataframe(un_add.head(200), use_container_width=True)
                st.download_button(
                    f"导出未分类单加（{sid}）CSV",
                    data=un_add.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"未分类单加_{sid}.csv",
                    mime="text/csv",
                )

    # ⑨ 明细导出
    with tabs[8]:
        st.subheader("明细导出：总部/财务/店长二次分析")

        for sid in sel_stores:
            st.markdown(f"#### 门店 {sid}")
            m = filtered[sid]["items_main"]
            a = filtered[sid]["items_add"]
            p = filtered[sid]["pay"]
            o = filtered[sid]["orders"]

            if not m.empty:
                st.download_button(
                    f"导出菜品明细-过滤后（{sid}）CSV",
                    data=m.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"菜品明细_过滤后_{sid}.csv",
                    mime="text/csv",
                )
            if not a.empty:
                st.download_button(
                    f"导出单加明细-过滤后（{sid}）CSV",
                    data=a.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"单加明细_过滤后_{sid}.csv",
                    mime="text/csv",
                )
            if not p.empty:
                st.download_button(
                    f"导出支付明细-过滤后（{sid}）CSV",
                    data=p.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"支付明细_过滤后_{sid}.csv",
                    mime="text/csv",
                )
            if not o.empty:
                st.download_button(
                    f"导出订单汇总-过滤后（{sid}）CSV",
                    data=o.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"订单汇总_过滤后_{sid}.csv",
                    mime="text/csv",
                )

    # ⑩ 时段热力图
    with tabs[9]:
        st.subheader("时段热力图（30分钟粒度）：峰谷、排班、备货、渠道动作")
        st.caption("行=日期，列=半小时；可选择指标；支持选中门店汇总或按门店分别查看。")

        metric = st.selectbox(
            "选择指标",
            options=["订单数", "应收(优惠后)", "实收", "客单(应收/订单)", "单加渗透率(含单加订单/总订单)"],
            index=0,
        )
        scope = st.radio("范围", options=["选中门店汇总", "按门店分别看"], horizontal=True)

        def _build_heat(o_df: pd.DataFrame, p_df: pd.DataFrame, a_df: pd.DataFrame) -> Optional[pd.DataFrame]:
            if o_df is None or o_df.empty:
                return None

            o = o_df.copy()
            o["date"] = o["order_time"].dt.date
            o["slot"] = o["order_time"].dt.floor("30min").dt.strftime("%H:%M")
            grp = o.groupby(["date", "slot"], as_index=False).agg(orders=("POS销售单号", "nunique"), net=("net_amount", "sum"))

            if p_df is not None and not p_df.empty:
                p = p_df.copy()
                p["date"] = p["order_time"].dt.date
                p["slot"] = p["order_time"].dt.floor("30min").dt.strftime("%H:%M")
                paid = p.groupby(["date", "slot"], as_index=False).agg(paid=("总金额", "sum"))
                grp = grp.merge(paid, on=["date", "slot"], how="left")
            grp["paid"] = grp.get("paid", 0).fillna(0.0)

            if a_df is not None and not a_df.empty:
                a = a_df.copy()
                a["date"] = a["created_at"].dt.date
                a["slot"] = a["created_at"].dt.floor("30min").dt.strftime("%H:%M")
                add_o = a.groupby(["date", "slot"], as_index=False).agg(add_orders=("order_id", "nunique"))
                grp = grp.merge(add_o, on=["date", "slot"], how="left")
            grp["add_orders"] = grp.get("add_orders", 0).fillna(0)

            grp["aov"] = grp["net"] / grp["orders"].replace(0, np.nan)
            grp["add_rate"] = grp["add_orders"] / grp["orders"].replace(0, np.nan)
            return grp

        def _render_heat(df: Optional[pd.DataFrame]) -> None:
            if df is None or df.empty:
                st.info("无可用数据")
                return

            if metric == "订单数":
                mat = df.pivot(index="date", columns="slot", values="orders").fillna(0).astype(int)
            elif metric == "应收(优惠后)":
                mat = df.pivot(index="date", columns="slot", values="net").fillna(0.0)
            elif metric == "实收":
                mat = df.pivot(index="date", columns="slot", values="paid").fillna(0.0)
            elif metric == "客单(应收/订单)":
                mat = df.pivot(index="date", columns="slot", values="aov")
            else:
                mat = df.pivot(index="date", columns="slot", values="add_rate")

            cols = sorted(mat.columns, key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
            mat = mat[cols]

            st.dataframe(mat.style.background_gradient(axis=None), use_container_width=True)

            flat = df.copy()
            if metric == "订单数":
                flat["v"] = flat["orders"]
            elif metric == "应收(优惠后)":
                flat["v"] = flat["net"]
            elif metric == "实收":
                flat["v"] = flat["paid"]
            elif metric == "客单(应收/订单)":
                flat["v"] = flat["aov"]
            else:
                flat["v"] = flat["add_rate"]

            top = flat.sort_values("v", ascending=False).head(10)[["date", "slot", "v", "orders", "net", "paid", "add_rate"]]
            st.markdown("**Top10 时段（按当前指标）**")
            st.dataframe(top, use_container_width=True)

        if scope == "选中门店汇总":
            oall = pd.concat([filtered[s]["orders"] for s in sel_stores], ignore_index=True)
            pall = pd.concat([filtered[s]["pay"] for s in sel_stores], ignore_index=True)
            aall = pd.concat([filtered[s]["items_add"] for s in sel_stores], ignore_index=True)
            _render_heat(_build_heat(oall, pall, aall))
        else:
            for sid in sel_stores:
                st.markdown(f"#### 门店 {sid}")
                _render_heat(_build_heat(filtered[sid]["orders"], filtered[sid]["pay"], filtered[sid]["items_add"]))


if __name__ == "__main__":
    main()
