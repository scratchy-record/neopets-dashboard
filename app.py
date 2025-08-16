# app.py — Neopets Wishlist Dashboard (Local Streamlit App)
# ---------------------------------------------------------
# - Reads Google Sheets tabs: prices_long + summary_30d via CSV export
# - Filters by item title, category, rarity
# - Shows Top Movers, Top Decliners, Highest Volatility
# - Line chart for selected items
#
# Install once:
#   pip3 install streamlit pandas plotly requests python-dateutil
#
# Run:
#   streamlit run app.py
# ---------------------------------------------------------

import io
import re
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Neopets Wishlist Dashboard", layout="wide")

PRICES_TAB  = "prices_long"
SUMMARY_TAB = "summary_30d"  # you’re using this tab name for all-time summary

# ---------- Utilities ----------
def sheet_id_from_url(sheet_url: str) -> str:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if not m:
        raise ValueError("Could not extract Sheet ID from the provided URL.")
    return m.group(1)

def csv_url_for_tab(sheet_url: str, tab_name: str) -> str:
    sid = sheet_id_from_url(sheet_url)
    return f"https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv&sheet={requests.utils.quote(tab_name)}"

@st.cache_data(ttl=300, show_spinner=False)
def load_tab_as_df(sheet_url: str, tab_name: str) -> pd.DataFrame:
    url = csv_url_for_tab(sheet_url, tab_name)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def coerce_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "run_date" in out.columns:
        out["run_date"] = pd.to_datetime(out["run_date"], errors="coerce", utc=True).dt.tz_convert(None)
    if "item_id" in out.columns:
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
    if "rarity" in out.columns:
        out["rarity"] = pd.to_numeric(out["rarity"], errors="coerce").astype("Int64")
    if "current_price_np" in out.columns:
        out["current_price_np"] = pd.to_numeric(out["current_price_np"], errors="coerce")
    if "item_category" not in out.columns:
        out["item_category"] = "Unknown"
    return out

def coerce_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num_cols = ["price_earliest","price_latest","abs_change","pct_change","volatility_pct","rarity","target_price"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    date_cols = ["date_earliest","date_latest"]
    for c in date_cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce", utc=True).dt.tz_convert(None)
    if "item_id" in out.columns:
        out["item_id"] = pd.to_numeric(out["item_id"], errors="coerce").astype("Int64")
    if "item_category" not in out.columns:
        out["item_category"] = "Unknown"
    return out

def build_alltime_from_prices(df_prices: pd.DataFrame) -> pd.DataFrame:
    g = df_prices.dropna(subset=["item_id","run_date","current_price_np"]).sort_values(["item_id","run_date"]).copy()
    if g.empty:
        return pd.DataFrame(columns=[
            "item_id","item_name","rarity","item_category","link",
            "date_earliest","price_earliest","date_latest","price_latest",
            "abs_change","pct_change","volatility_pct"
        ])
    earliest = g.groupby("item_id").first().reset_index()
    latest   = g.groupby("item_id").last().reset_index()
    base = g[["item_id","item_name","rarity","item_category","link"]].drop_duplicates("item_id")
    vol = (
        g.groupby("item_id")["current_price_np"]
         .apply(lambda s: s.pct_change().dropna().std() * 100 if s.notna().any() else np.nan)
         .reset_index(name="volatility_pct")
    )
    s = (
        base
        .merge(earliest[["item_id","run_date","current_price_np"]]
               .rename(columns={"run_date":"date_earliest","current_price_np":"price_earliest"}), on="item_id", how="left")
        .merge(latest[["item_id","run_date","current_price_np"]]
               .rename(columns={"run_date":"date_latest","current_price_np":"price_latest"}), on="item_id", how="left")
        .merge(vol, on="item_id", how="left")
    )
    s["abs_change"] = s["price_latest"] - s["price_earliest"]
    s["pct_change"] = np.where(
        s["price_earliest"] > 0, (s["abs_change"] / s["price_earliest"]) * 100, np.nan
    )
    return s

def top_tables(summary_df: pd.DataFrame, n=20, title_filter="", cat_filter=None, rarity_filter=None):
    dfv = summary_df.copy()
    if title_filter:
        q = title_filter.lower()
        dfv = dfv[dfv["item_name"].str.lower().str.contains(q, na=False)]
    if cat_filter:
        dfv = dfv[dfv["item_category"].isin(cat_filter)]
    if rarity_filter:
        dfv = dfv[dfv["rarity"].isin(rarity_filter)]
    gainers   = dfv.sort_values("pct_change", ascending=False).head(n)
    decliners = dfv.sort_values("pct_change", ascending=True).head(n)
    volatile  = dfv.sort_values("volatility_pct", ascending=False).head(n)
    return gainers, decliners, volatile

def apply_filters(df, title_filter, cat_filter, rarity_filter):
    dfv = df.copy()
    if title_filter:
        q = title_filter.lower()
        dfv = dfv[dfv["item_name"].str.lower().str.contains(q, na=False)]
    if cat_filter:
        dfv = dfv[dfv["item_category"].isin(cat_filter)]
    if rarity_filter:
        dfv = dfv[dfv["rarity"].isin(rarity_filter)]
    return dfv

# Build alerts from filtered summary
summary_filtered = apply_filters(summary, title_filter, cat_filter, rarity_filter)
alerts = summary_filtered.copy()
if "target_price" in alerts.columns:
    alerts = alerts[alerts["target_price"].notna()]
    alerts = alerts[alerts["price_latest"].notna()]
    alerts = alerts[alerts["price_latest"] >= alerts["target_price"]]
    # Nice ordering: how far over target
    alerts["over_target"] = alerts["price_latest"] - alerts["target_price"]
    alerts = alerts.sort_values(["over_target","pct_change"], ascending=[False, False])
else:
    alerts = alerts.iloc[0:0]  # empty if no column in sheet

def plot_lines(prices: pd.DataFrame, item_ids: list[int]):
    sub = prices[prices["item_id"].isin(item_ids)].dropna(subset=["run_date","current_price_np"]).copy()
    if sub.empty:
        st.info("No price data to plot for the current selection.")
        return
    sub["legend"] = sub.apply(lambda r: f"{r['item_name']} ({r['item_id']})", axis=1)
    fig = px.line(
        sub.sort_values("run_date"),
        x="run_date", y="current_price_np", color="legend",
        labels={"run_date":"Date","current_price_np":"Price (NP)","legend":"Item"},
        title="Price History"
    )
    fig.update_layout(hovermode="x unified", legend_title_text="Item")
    st.plotly_chart(fig, use_container_width=True)

# ---------- UI: Sidebar ----------
st.title("🧪 Neopets Wishlist Dashboard (Local)")
st.caption("Reads Google Sheet tabs: prices_long + summary_30d (all-time).")

with st.sidebar:
    st.header("Data Source")
    SHEET_URL = st.text_input(
        "Google Sheet URL",
        value="",
        help=(
            "Paste the full Google Sheet URL (with /spreadsheets/d/<ID>/...).\n"
            "Share the sheet as 'Anyone with the link: Viewer' (or Publish to web) so the CSV export works."
        ),
    )
    st.markdown(
        f"- Expects tabs: **{PRICES_TAB}** and **{SUMMARY_TAB}**\n"
        "- If **{SUMMARY_TAB}** is missing, the app will compute an all-time summary from prices_long."
    )
    reload_btn = st.button("Reload data")

if not SHEET_URL:
    st.info("Paste your Google Sheet URL in the sidebar to begin.")
    st.stop()

# ---------- Load data ----------
try:
    prices = load_tab_as_df(SHEET_URL, PRICES_TAB)
except Exception as e:
    st.error(f"Failed to load '{PRICES_TAB}': {e}")
    st.stop()

prices = coerce_prices(prices)

# Try to load summary; if missing or fails, compute from prices
try:
    summary = load_tab_as_df(SHEET_URL, SUMMARY_TAB)
    summary = coerce_summary(summary)
    if summary.empty:
        summary = build_alltime_from_prices(prices)
except Exception:
    summary = build_alltime_from_prices(prices)

# ---------- Filters ----------
all_categories = sorted(prices["item_category"].fillna("Unknown").unique().tolist()) if "item_category" in prices.columns else []
all_rarities = sorted([int(x) for x in prices["rarity"].dropna().unique().tolist()]) if "rarity" in prices.columns else []

colA, colB, colC, colD = st.columns([2,2,2,2])
with colA:
    title_filter = st.text_input("Filter by Item Title (contains)", "")
with colB:
    cat_filter = st.multiselect("Filter by Category", options=all_categories, default=[])
with colC:
    rarity_filter = st.multiselect("Filter by Rarity", options=all_rarities, default=[])
with colD:
    top_n = st.slider("Top N", min_value=5, max_value=100, value=20, step=5)

# ---------- Tables & Charts ----------
gainers, decliners, volatile = top_tables(summary, top_n, title_filter, cat_filter, rarity_filter)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Top Movers", "📉 Top Decliners", "⚡ Highest Volatility", "📊 Line Chart", "🔔 Price Alerts"])
with tab5:
    st.subheader("Price Alerts (price_latest ≥ target_price)")
    if alerts.empty:
        st.info("No alerts. Add 'target_price' values in the summary_30d sheet, or adjust filters.")
    else:
        cols = ["item_name","link","price_earliest","price_latest","pct_change","target_price"]
        # Only include columns that exist (robust if sheet is missing any)
        cols = [c for c in cols if c in alerts.columns]
        st.dataframe(alerts[cols].reset_index(drop=True))
with tab1:
    st.subheader("Top Movers (All-time % change)")
    cols = ["item_name","item_id","item_category","rarity","price_earliest","price_latest","abs_change","pct_change","link"]
    st.dataframe(gainers[cols].reset_index(drop=True))

with tab2:
    st.subheader("Top Decliners (All-time % change)")
    cols = ["item_name","item_id","item_category","rarity","price_earliest","price_latest","abs_change","pct_change","link"]
    st.dataframe(decliners[cols].reset_index(drop=True))

with tab3:
    st.subheader("Highest Volatility (All-time std dev of % returns)")
    cols = ["item_name","item_id","item_category","rarity","volatility_pct","link"]
    st.dataframe(volatile[cols].reset_index(drop=True))

with tab4:
    # Default to top 5 gainers
    default_ids = gainers["item_id"].head(5).dropna().astype(int).tolist()
    # Choices filtered by current filters
    filtered_prices = prices.copy()
    if title_filter:
        q = title_filter.lower()
        filtered_prices = filtered_prices[filtered_prices["item_name"].str.lower().str.contains(q, na=False)]
    if cat_filter:
        filtered_prices = filtered_prices[filtered_prices["item_category"].isin(cat_filter)]
    if rarity_filter:
        filtered_prices = filtered_prices[filtered_prices["rarity"].isin(rarity_filter)]
    choices = filtered_prices[["item_id","item_name"]].drop_duplicates().sort_values("item_name")
    label_map = {int(r.item_id): f"{r.item_name} ({int(r.item_id)})" for _, r in choices.iterrows() if pd.notna(r.item_id)}
    select_ids = st.multiselect(
        "Select items to plot",
        options=list(label_map.keys()),
        default=default_ids,
        format_func=lambda k: label_map.get(k, str(k))
    )
    plot_lines(prices, select_ids)
    
