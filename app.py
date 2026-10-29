"""Blinkit Quick Commerce Sales and Predictive Analytics - Streamlit dashboard."""
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Blinkit Quick Commerce Analytics", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

GREEN, EMERALD, YELLOW, CORAL, MUTED, GRID = "#006B3C", "#00874E", "#F8CB46", "#E57373", "#5F6B63", "#E8EFEA"
PALETTE = [GREEN, YELLOW, EMERALD, CORAL, "#4DA377", "#B8860B", "#90CAF9"]

st.markdown(f"""
<style>
.stApp {{ background:#F3F4F0; }}
section[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid #D9E6DE; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#FFFFFF; border-radius:10px; }}
.hdr {{ background:#FFFFFF; border:1px solid #D9E6DE; border-radius:10px; padding:16px 22px; margin-bottom:16px; }}
.hdr h2 {{ color:{GREEN}; margin:0; font-size:23px; font-weight:800; }}
.hdr p {{ color:{MUTED}; margin:4px 0 0; font-size:13px; }}
.kpi {{ background:#FFFFFF; border:1px solid #D9E6DE; border-left:5px solid {YELLOW}; border-radius:10px; padding:14px 18px; }}
.kpi .v {{ color:{GREEN}; font-size:27px; font-weight:800; line-height:1.1; }}
.kpi .t {{ color:#1E2D24; font-size:13px; font-weight:600; margin-top:3px; }}
.kpi .s {{ color:{MUTED}; font-size:11px; margin-top:2px; }}
.ctitle {{ color:{GREEN}; font-size:15px; font-weight:700; margin-bottom:4px; }}
.matrix {{ padding:12px 16px; border-radius:8px; margin-bottom:10px; font-size:13px; line-height:1.6; }}
</style>""", unsafe_allow_html=True)

# ---------- version-safe helpers ----------
_STRETCH = "width" in inspect.signature(st.plotly_chart).parameters
_W = {"width": "stretch"} if _STRETCH else {"use_container_width": True}
_DF_STRETCH = "width" in inspect.signature(st.dataframe).parameters
_DW = {"width": "stretch"} if _DF_STRETCH else {"use_container_width": True}


def show(fig, h=320, key=None):
    fig.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", height=h,
                      margin=dict(l=10, r=10, t=10, b=10), font=dict(color="#1E2D24"),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, key=key, config={"displaylogo": False}, **_W)


def kpis(items):
    for col, (v, t, s) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="t">{t}</div>'
                     f'<div class="s">{s}</div></div>', unsafe_allow_html=True)
    st.write("")


def header(title, sub):
    st.markdown(f'<div class="hdr"><h2>{title}</h2><p>{sub}</p></div>', unsafe_allow_html=True)


def card(title):
    box = st.container(border=True)
    box.markdown(f'<div class="ctitle">{title}</div>', unsafe_allow_html=True)
    return box


def pct(a, b):
    return a / b * 100 if b else 0.0


def inr(x):
    return f"₹{x / 1e6:.2f}M" if abs(x) >= 1e6 else f"₹{x / 1e5:.2f}L" if abs(x) >= 1e5 else f"₹{x:,.0f}"


# ---------- data ----------
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    base = Path(__file__).resolve().parent
    files = sorted(p for p in base.rglob("*.csv") if ".git" not in p.parts)
    if not files:
        return None, {}
    path = next((p for p in files if "master" in p.name.lower()), files[0])
    raw = pd.read_csv(path)
    rep = {"file": path.name, "rows_loaded": len(raw), "columns": raw.shape[1]}
    df = raw.copy()
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("string").str.strip()
    df = df.drop_duplicates()
    rep["exact_duplicates_removed"] = rep["rows_loaded"] - len(df)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    rep["bad_dates_removed"] = int(df["order_date"].isna().sum())
    df = df.dropna(subset=["order_date"])
    for c in ["order_total", "item_revenue", "total_spending", "average_order_value", "total_orders",
              "repeat_probability", "delay_rate", "recency_days"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    rep["rows_final"] = len(df)
    rep["missing_cells"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), rep


df, report = load_data()
if df is None:
    st.error("No CSV found next to app.py. Add the master dataset to the repository.")
    st.stop()
missing = [c for c in ["order_date", "customer_id", "order_total"] if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}")
    st.stop()
has = lambda *cols: all(c in df.columns for c in cols)  # noqa: E731

# ---------- sidebar ----------
st.sidebar.markdown(f'<h2 style="color:{GREEN};margin:0;font-weight:800">blink<span style="color:{YELLOW}">it</span></h2>'
                    f'<p style="color:{MUTED};font-size:11px;font-weight:600;margin:0 0 12px">QUICK COMMERCE ANALYTICS</p>',
                    unsafe_allow_html=True)
PAGES = ["Executive Overview", "Customer Intelligence", "Predictive Intelligence",
         "Delivery & Retention", "Data Explorer"]
page = st.sidebar.radio("Dashboard", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown("**Filters**")

dmin, dmax = df["order_date"].dt.date.min(), df["order_date"].dt.date.max()
FILTERS = {"f_cat": "category", "f_seg": "customer_segment", "f_delay": "delay_flag", "f_pay": "payment_method"}


def reset_filters():
    st.session_state["f_date"] = (dmin, dmax)
    for k, col in FILTERS.items():
        if col in df.columns:
            st.session_state[k] = sorted(df[col].dropna().unique().tolist())


drange = st.sidebar.date_input("Order date", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="f_date")
mask = pd.Series(True, index=df.index)
start, end = (drange[0], drange[-1]) if isinstance(drange, (tuple, list)) and len(drange) else (dmin, dmax)
mask &= (df["order_date"].dt.date >= start) & (df["order_date"].dt.date <= end)
LABELS = {"f_cat": "Category", "f_seg": "Customer segment", "f_delay": "Delivery SLA", "f_pay": "Payment method"}
for k, col in FILTERS.items():
    if col in df.columns:
        opts = sorted(df[col].dropna().unique().tolist())
        sel = st.sidebar.multiselect(LABELS[k], opts, default=opts, key=k)
        mask &= df[col].isin(sel)
st.sidebar.button("Reset filters", on_click=reset_filters)

fdf = df[mask]
if fdf.empty:
    header("No data", "No records match the current filters.")
    st.warning("Widen the filters in the sidebar or press **Reset filters**.")
    st.stop()

# order-level frame (one row per order) so sales are never double counted
orders = fdf.drop_duplicates("order_id") if "order_id" in fdf.columns else fdf
cust = fdf.drop_duplicates("customer_id")
st.sidebar.caption(f"{len(orders):,} orders · {len(cust):,} customers in view")

# =====================================================================
if page == "Executive Overview":
    header("Executive Overview", "Revenue, order volume and basket size at a glance")
    sales, n_ord = orders["order_total"].sum(), len(orders)
    kpis([(f"{cust['customer_id'].nunique():,}", "Customers", "Unique buyers in view"),
          (f"{n_ord:,}", "Orders", "Unique orders"),
          (inr(sales), "Total sales", f"GMV ₹{sales:,.0f}"),
          (f"₹{sales / n_ord:,.0f}", "Avg order value", "Sales ÷ orders")])

    c1, c2 = st.columns([1.15, 1])
    with c1, card("Sales trend"):
        a, b = st.columns(2)
        grain = a.radio("Group by", ["Month", "Week", "Day"], horizontal=True, key="grain")
        metric = b.radio("Metric", ["Sales", "Orders", "AOV"], horizontal=True, key="metric")
        freq = {"Month": "MS", "Week": "W-MON", "Day": "D"}[grain]
        t = orders.set_index("order_date").resample(freq).agg(Sales=("order_total", "sum"), Orders=("order_total", "size"))
        t["AOV"] = (t["Sales"] / t["Orders"]).replace([np.inf], np.nan)
        fig = go.Figure(go.Scatter(x=t.index, y=t[metric], mode="lines+markers", line=dict(color=GREEN, width=3),
                                   marker=dict(color=YELLOW, size=6, line=dict(color=GREEN, width=1)),
                                   fill="tozeroy", fillcolor="rgba(0,107,60,0.08)",
                                   hovertemplate="%{x|%d %b %Y}<br>" + metric + ": %{y:,.0f}<extra></extra>"))
        show(fig, 300, "trend")
    with c2, card("Revenue by category"):
        if has("category"):
            val = "item_revenue" if has("item_revenue") else "order_total"
            topn = st.slider("Show top N", 3, max(3, fdf["category"].nunique()), min(8, max(3, fdf["category"].nunique())), key="topn")
            g = fdf.groupby("category")[val].sum().nlargest(topn).sort_values().reset_index()
            fig = px.bar(g, x=val, y="category", orientation="h", color_discrete_sequence=[GREEN])
            fig.update_traces(hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>")
            fig.update_layout(xaxis_title="Revenue (₹)", yaxis_title="")
            show(fig, 300, "cat")
        else:
            st.info("No category column in dataset.")
    c3, c4 = st.columns(2)
    with c3, card("Payment method mix"):
        if has("payment_method"):
            g = orders["payment_method"].value_counts().reset_index()
            g.columns = ["Method", "Orders"]
            fig = px.pie(g, names="Method", values="Orders", hole=0.55, color_discrete_sequence=PALETTE)
            show(fig, 280, "pay")
        else:
            st.info("No payment_method column in dataset.")
    with c4, card("Orders by weekday"):
        w = orders["order_date"].dt.day_name().value_counts()
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        w = w.reindex(days, fill_value=0).reset_index()
        w.columns = ["Day", "Orders"]
        fig = px.bar(w, x="Day", y="Orders", color_discrete_sequence=[EMERALD])
        show(fig, 280, "wk")

# =====================================================================
elif page == "Customer Intelligence":
    header("Customer Intelligence", "Segments, repeat behaviour and spending profiles")
    need = ["total_orders", "total_spending", "customer_segment"]
    if not has(*need):
        st.error(f"Required columns missing: {', '.join(c for c in need if not has(c))}")
        st.stop()
    n = len(cust)
    rep = int((cust["total_orders"] > 1).sum())
    kpis([(f"{n:,}", "Customers", "In current view"),
          (f"{pct(rep, n):.1f}%", "Repeat rate", f"{rep:,} repeat shoppers"),
          (f"{n - rep:,}", "One-time buyers", f"{pct(n - rep, n):.1f}% of customers"),
          (f"{cust['customer_segment'].nunique()}", "Segments", "Behavioural clusters")])

    c1, c2 = st.columns([1.2, 1])
    with c1, card("Segment comparison"):
        opts = {"Total spending": ("total_spending", "sum"), "Avg order value": ("average_order_value", "mean"),
                "Avg orders per customer": ("total_orders", "mean"), "Customers": ("customer_id", "nunique")}
        opts = {k: v for k, v in opts.items() if v[0] in cust.columns}
        pick = st.selectbox("Metric", list(opts), key="segmetric")
        col, agg = opts[pick]
        g = cust.groupby("customer_segment")[col].agg(agg).reset_index().sort_values(col, ascending=False)
        fig = px.bar(g, x="customer_segment", y=col, color="customer_segment", color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title=pick)
        show(fig, 300, "seg")
    with c2, card("Customer share by segment"):
        g = cust["customer_segment"].value_counts().reset_index()
        g.columns = ["Segment", "Customers"]
        show(px.pie(g, names="Segment", values="Customers", hole=0.55, color_discrete_sequence=PALETTE), 300, "segpie")

    c3, c4 = st.columns([1, 1.2])
    with c3, card("Spending vs order frequency"):
        s = cust.sample(min(len(cust), 3000), random_state=1)
        hd = [c for c in ["customer_id", "average_order_value"] if c in s.columns]
        fig = px.scatter(s, x="total_orders", y="total_spending", color="customer_segment",
                         color_discrete_sequence=PALETTE, hover_data=hd, opacity=0.75)
        show(fig, 340, "scatter")
    with c4, card("Top spenders"):
        q = st.text_input("Search customer ID or name", key="custq")
        cols = [c for c in ["customer_id", "customer_name", "total_spending", "total_orders", "average_order_value",
                            "customer_segment", "repeat_prediction_label"] if c in cust.columns]
        t = cust[cols]
        if q:
            m = t["customer_id"].astype(str).str.contains(q, case=False, na=False)
            if "customer_name" in t.columns:
                m |= t["customer_name"].astype(str).str.contains(q, case=False, na=False)
            t = t[m]
        st.dataframe(t.sort_values("total_spending", ascending=False).head(15), hide_index=True,
                     column_config={"total_spending": st.column_config.NumberColumn("Total spend", format="₹%.0f"),
                                    "average_order_value": st.column_config.NumberColumn("AOV", format="₹%.0f")}, **_DW)

# =====================================================================
elif page == "Predictive Intelligence":
    header("Predictive Intelligence", "Repeat-purchase propensity and the actions it suggests")
    need = ["repeat_prediction_label", "repeat_probability", "customer_segment"]
    if not has(*need):
        st.error(f"Required columns missing: {', '.join(c for c in need if not has(c))}")
        st.stop()
    n = len(cust)
    likely = int((cust["repeat_prediction_label"] == "Likely to Repeat").sum())
    kpis([(f"{likely:,}", "Likely to repeat", f"{pct(likely, n):.1f}% of customers"),
          (f"{n - likely:,}", "Unlikely to repeat", f"{pct(n - likely, n):.1f}% at risk"),
          (inr(cust["total_spending"].sum()) if has("total_spending") else "n/a", "Cohort spend", "Lifetime tracked spend"),
          (f"{cust['repeat_probability'].mean() * 100:.1f}%", "Avg repeat probability", "Model score")])

    c1, c2, c3 = st.columns(3)
    with c1, card("Repeat prediction"):
        g = cust["repeat_prediction_label"].value_counts().reset_index()
        g.columns = ["Label", "Customers"]
        show(px.pie(g, names="Label", values="Customers", hole=0.55, color="Label",
                    color_discrete_map={"Likely to Repeat": GREEN, "Unlikely to Repeat": CORAL}), 270, "pred")
    with c2, card("Probability distribution"):
        show(px.histogram(cust, x="repeat_probability", nbins=20, color_discrete_sequence=[GREEN]), 270, "hist")
    with c3, card("Recommended actions"):
        if has("loyalty_recommendation"):
            g = cust["loyalty_recommendation"].value_counts().reset_index()
            g.columns = ["Action", "Customers"]
            show(px.bar(g, x="Customers", y="Action", orientation="h", color_discrete_sequence=[YELLOW]), 270, "rec")
        else:
            st.info("No loyalty_recommendation column.")

    with card("Intervention matrix (live counts)"):
        thr = st.slider("High-propensity threshold", 0.1, 0.9, 0.5, 0.05, key="thr")
        hv = (cust["customer_segment"] == "High-Value Customers")
        hp = cust["repeat_probability"] >= thr
        spend = cust["total_spending"] if has("total_spending") else pd.Series(0, index=cust.index)
        boxes = [("#E8F5E9", "#006B3C", "High value + high propensity → VIP loyalty club", hv & hp),
                 ("#FFF8E1", "#B8860B", "Other segments + high propensity → basket acceleration", ~hv & hp),
                 ("#FFEBEE", "#C62828", "High value + low propensity → priority win-back", hv & ~hp),
                 ("#E3F2FD", "#1565C0", "Other segments + low propensity → light-touch nurture", ~hv & ~hp)]
        a, b = st.columns(2)
        for i, (bg, fg, title, m) in enumerate(boxes):
            (a if i % 2 == 0 else b).markdown(
                f'<div class="matrix" style="background:{bg};border:1px solid {fg}33">'
                f'<b style="color:{fg}">{title}</b><br>{int(m.sum()):,} customers · {inr(spend[m].sum())} spend</div>',
                unsafe_allow_html=True)

    with card("Priority roster: high-value customers at risk"):
        risk = cust[hv & ~hp]
        cols = [c for c in ["customer_id", "customer_name", "total_spending", "total_orders", "delay_rate",
                            "recency_days", "repeat_probability", "loyalty_recommendation",
                            "personalised_offer_recommendation"] if c in risk.columns]
        if risk.empty:
            st.info("No high-value at-risk customers for the current filters and threshold.")
        else:
            r = risk[cols].sort_values("total_spending" if "total_spending" in cols else cols[0], ascending=False)
            st.dataframe(r, hide_index=True, **_DW)
            st.download_button("Download roster (CSV)", r.to_csv(index=False).encode(), "high_value_at_risk.csv", "text/csv")

# =====================================================================
elif page == "Delivery & Retention":
    header("Delivery & Retention", "SLA performance, delay causes and the link to repeat buying")
    if not has("delay_flag"):
        st.error("Required column missing: delay_flag")
        st.stop()
    n = len(orders)
    delayed = int((orders["delay_flag"] == "Delayed").sum())
    rr = pct(int((cust["total_orders"] > 1).sum()), len(cust)) if has("total_orders") else float("nan")
    kpis([(f"{n:,}", "Orders", "Analysed"), (f"{delayed:,}", "Delayed orders", "Past promised SLA"),
          (f"{pct(delayed, n):.1f}%", "Delay rate", "Delayed ÷ orders"),
          (f"{rr:.1f}%" if rr == rr else "n/a", "Repeat purchase rate", "Customer level")])

    c1, c2, c3 = st.columns(3)
    with c1, card("On time vs delayed"):
        g = orders["delay_flag"].value_counts().reset_index()
        g.columns = ["Status", "Orders"]
        show(px.pie(g, names="Status", values="Orders", hole=0.55, color="Status",
                    color_discrete_map={"On Time": GREEN, "Delayed": CORAL}), 270, "dl")
    with c2, card("Delay root causes"):
        if has("reasons_if_delayed"):
            r = orders[(orders["delay_flag"] == "Delayed") & (orders["reasons_if_delayed"] != "Not Delayed")]
            g = r["reasons_if_delayed"].value_counts().reset_index()
            g.columns = ["Reason", "Orders"]
            if g.empty:
                st.info("No delayed orders in view.")
            else:
                show(px.bar(g.sort_values("Orders"), x="Orders", y="Reason", orientation="h",
                            color_discrete_sequence=[GREEN]), 270, "rs")
        else:
            st.info("No reasons_if_delayed column.")
    with c3, card("Delay vs repeat propensity"):
        if has("repeat_prediction_label"):
            g = orders.groupby(["delay_flag", "repeat_prediction_label"]).size().reset_index(name="Orders")
            g["Share %"] = g["Orders"] / g.groupby("delay_flag")["Orders"].transform("sum") * 100
            show(px.bar(g, x="delay_flag", y="Share %", color="repeat_prediction_label", barmode="stack",
                        color_discrete_map={"Likely to Repeat": GREEN, "Unlikely to Repeat": CORAL}), 270, "dr")
        else:
            st.info("No repeat_prediction_label column.")

    with card("Findings (calculated from the filtered data)"):
        pts = [f"**SLA breaches:** {pct(delayed, n):.1f}% of {n:,} orders were delayed."]
        if has("reasons_if_delayed") and delayed:
            top = orders[orders["delay_flag"] == "Delayed"]["reasons_if_delayed"].value_counts()
            if len(top):
                pts.append(f"**Top cause:** {top.index[0]} ({pct(top.iloc[0], delayed):.1f}% of delayed orders).")
        if has("delay_rate", "total_orders"):
            hi, lo = cust[cust["delay_rate"] > 0.5], cust[cust["delay_rate"] <= 0.5]
            if len(hi) and len(lo):
                pts.append(f"**Retention link:** customers with >50% delayed orders repeat at "
                           f"{pct(int((hi['total_orders'] > 1).sum()), len(hi)):.1f}% vs "
                           f"{pct(int((lo['total_orders'] > 1).sum()), len(lo)):.1f}% for the rest.")
        st.markdown("\n".join(f"- {p}" for p in pts))

# =====================================================================
else:
    header("Data Explorer", "Data-quality report, column picker and exports")
    kpis([(f"{report['rows_loaded']:,}", "Rows loaded", report["file"]),
          (f"{report['exact_duplicates_removed']:,}", "Duplicate rows removed", "Exact duplicates"),
          (f"{report['rows_final']:,}", "Rows after cleaning", f"{report['columns']} columns"),
          (f"{report['missing_cells']:,}", "Missing cells", "After cleaning")])
    with card("Missing values by column"):
        mv = df.isna().sum()
        mv = mv[mv > 0].sort_values(ascending=False).reset_index()
        mv.columns = ["Column", "Missing"]
        st.success("No missing values.") if mv.empty else st.dataframe(mv, hide_index=True, **_DW)
    with card("Preview and export"):
        scope = st.radio("Data", ["Filtered view", "Full dataset"], horizontal=True)
        base = fdf if scope == "Filtered view" else df
        cols = st.multiselect("Columns", list(df.columns), default=list(df.columns[:12]))
        view = base[cols] if cols else base
        st.caption(f"{len(view):,} rows")
        st.dataframe(view.head(500), hide_index=True, **_DW)
        st.download_button("Download CSV", view.to_csv(index=False).encode(), "blinkit_export.csv", "text/csv")
