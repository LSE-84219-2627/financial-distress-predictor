"""
app.py

Streamlit dashboard for the financial distress pipeline.

Run from the project root:
    streamlit run app.py

Reads everything from db/distress.db — no model refitting at runtime.
Tables used:
  annual_labeled  : accounting data, derived labels, available_from date
  oof_predictions : LR + XGB out-of-fold predicted probabilities (138 rows)
  companies       : cik -> ticker/name mapping
"""

import sqlite3
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent / "db" / "distress.db"

RATIO_COLS = [
    "leverage",           # liabilities / assets
    "equity_ratio",       # equity / assets
    "retained_to_assets", # retained_earnings / assets
    "roa",                # net_income / assets
    "margin",             # net_income / revenue
    "current_ratio",      # assets_current / liabilities_current
]

# ---------------------------------------------------------------------------
# Data loading — cached so the DB is only hit once per session
# ---------------------------------------------------------------------------

@st.cache_data
def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load annual_labeled and oof_predictions from SQLite.

    Ratio features are computed here (not stored in the DB) so the DB
    stays auditable (raw USD values) and feature definitions are version-
    controlled in this file.

    Returns (annual, oof) where:
      annual  — 154-row table, all fiscal years including ones without a
                t+1 target.  Contains raw financials, derived labels,
                and ratio features.
      oof     — 138-row table, only years with a distressed_next label.
                Contains lr_prob and xgb_prob from LOFO cross-validation.
    """
    conn = sqlite3.connect(DB_PATH)

    annual = pd.read_sql(
        "SELECT * FROM annual_labeled", conn,
        parse_dates=["filed", "available_from"],
    )
    oof = pd.read_sql("SELECT cik, ticker, fy, lr_prob, xgb_prob FROM oof_predictions", conn)

    conn.close()

    # Derive ratio features (propagates NaN correctly when operand is NaN)
    annual["leverage"]           = annual["liabilities"]       / annual["assets"]
    annual["equity_ratio"]       = annual["equity"]            / annual["assets"]
    annual["retained_to_assets"] = annual["retained_earnings"] / annual["assets"]
    annual["roa"]                = annual["net_income"]        / annual["assets"]
    annual["margin"]             = annual["net_income"]        / annual["revenue"]
    annual["current_ratio"]      = annual["assets_current"]    / annual["liabilities_current"]

    return annual, oof


def firm_table(annual: pd.DataFrame, oof: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge annual rows with OOF predictions for one ticker.

    Rows where distressed_next is NaN (the last year of each firm, which has
    no t+1) are included in the display but will have lr_prob = NaN.
    """
    rows = annual[annual["ticker"] == ticker].copy()
    rows = rows.merge(
        oof[oof["ticker"] == ticker][["fy", "lr_prob", "xgb_prob"]],
        on="fy",
        how="left",
    )
    return rows.sort_values("fy").reset_index(drop=True)


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Distress Dashboard", layout="wide")

st.title("Financial Distress Dashboard")
st.caption(
    "10-firm panel · SEC EDGAR XBRL data · "
    "Labels: ICR < 1 (primary) or two-consecutive-loss-year fallback · "
    "Predicted probability: logistic regression, leave-one-firm-out CV"
)

annual, oof = load_all()
tickers     = sorted(annual["ticker"].unique())

# ---------------------------------------------------------------------------
# Sidebar: firm selector + methodology legend
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Select firm")
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("AMC"))

    st.markdown("---")
    st.markdown("### Distress label")
    st.markdown(
        "**coverage** — operating income < interest expense (ICR < 1).  "
        "Primary rule; both components must be non-null.\n\n"
        "**fallback** — net income < 0 in year *t* AND year *t−1*.  "
        "Used only when ICR components are missing."
    )
    st.markdown("### Predicted probability")
    st.markdown(
        "Out-of-fold from leave-one-firm-out CV.  "
        "The model **was never trained on this firm's data**, so the "
        "probability is a genuine out-of-sample estimate.  "
        "Target variable is `distressed_next` (distress at *t+1*)."
    )
    st.markdown("### Point-in-time")
    st.markdown(
        "`available_from` is the SEC filing date of the annual report.  "
        "Features are only observable after this date — no look-ahead."
    )

firm = firm_table(annual, oof, ticker)

# ---------------------------------------------------------------------------
# Firm-level summary strip
# ---------------------------------------------------------------------------

n_years      = len(firm)
n_distressed = int((firm["distressed"] == 1).sum())
n_labeled    = int(firm["distressed"].notna().sum())
n_with_prob  = int(firm["lr_prob"].notna().sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Fiscal years", n_years)
col2.metric("Distressed years", f"{n_distressed} / {n_labeled} labeled")
col3.metric("LR prob available", f"{n_with_prob} years")
col4.metric(
    "Distress rate",
    f"{n_distressed/n_labeled:.0%}" if n_labeled > 0 else "N/A",
)

st.markdown("---")

# ---------------------------------------------------------------------------
# (a) Annual ratio table
# ---------------------------------------------------------------------------

st.subheader("(a) Annual financial ratios")
st.caption(
    "All ratios derived from EDGAR XBRL filings.  "
    "NaN (shown as —) means the underlying tag was not available."
)

ratio_display = (
    firm[["fy"] + RATIO_COLS]
    .set_index("fy")
    .rename(columns={
        "leverage":           "leverage  (L/A)",
        "equity_ratio":       "eq_ratio  (E/A)",
        "retained_to_assets": "ret/assets",
        "roa":                "ROA  (NI/A)",
        "margin":             "margin  (NI/Rev)",
        "current_ratio":      "current ratio",
    })
)

def fmt_ratio(x):
    return f"{x:.3f}" if pd.notna(x) else "—"

st.dataframe(
    ratio_display.style.format(fmt_ratio),
    use_container_width=True,
    height=min(50 + 35 * len(firm), 550),
)

st.markdown("---")

# ---------------------------------------------------------------------------
# (b)+(c) Distress labels and LR predicted probability
# ---------------------------------------------------------------------------

st.subheader("(b) + (c) Distress labels and LR predicted probability")
st.caption(
    "`distressed` = label for year *t*.  "
    "`distressed_next` = label for year *t+1* (the model target).  "
    "`lr_prob` = P(distressed_next=1) from logistic regression LOFO CV.  "
    "Blank lr_prob = last firm-year (no t+1 target exists)."
)

label_display = firm[[
    "fy", "distressed", "label_source", "distressed_next",
    "lr_prob", "available_from",
]].copy()

label_display["distressed"]      = label_display["distressed"].apply(
    lambda x: "1 (distressed)" if x == 1 else ("0" if x == 0 else "—"))
label_display["distressed_next"] = label_display["distressed_next"].apply(
    lambda x: "1 (distressed)" if x == 1 else ("0" if x == 0 else "—"))
label_display["lr_prob"]         = label_display["lr_prob"].apply(
    lambda x: f"{x:.3f}" if pd.notna(x) else "—")
label_display["available_from"]  = label_display["available_from"].dt.strftime("%Y-%m-%d")

label_display = label_display.rename(columns={
    "fy":             "FY",
    "distressed":     "distressed (t)",
    "label_source":   "label source",
    "distressed_next":"distressed (t+1)",
    "lr_prob":        "LR prob",
    "available_from": "available from",
})

st.dataframe(
    label_display.set_index("FY"),
    use_container_width=True,
    height=min(50 + 35 * len(firm), 550),
)

st.markdown("---")

# ---------------------------------------------------------------------------
# (d) Line chart: LR predicted probability vs actual distressed_next
# ---------------------------------------------------------------------------
#
# Design: single y-axis 0-1 for lr_prob.  Actual distress shown as shaded
# vertical bands (red rule marks at full height) only for distressed_next=1
# years.  This avoids the dual-axis resolve_scale(y='independent') bug where
# a collapsed bar domain (e.g. all-zeros for AAPL) caused the lr_prob points
# to be mis-positioned on screen.

st.subheader("(d) LR predicted probability vs actual distress label over time")
st.caption(
    "**Red line + points** = LR predicted P(distressed next year) — out-of-fold, "
    "model never trained on this firm.  "
    "**Red shaded bands** = years where `distressed_next = 1`.  "
    "Bands and line share the same 0–1 y-axis."
)

chart_df = firm[["fy", "lr_prob", "distressed_next"]].copy()
chart_df = chart_df[chart_df["lr_prob"].notna()].copy()
chart_df["distressed_next"] = chart_df["distressed_next"].astype(float)
chart_df["fy_str"] = chart_df["fy"].astype(str)

# Separate frame: only the years where distressed_next=1, for the bands.
# Rule marks with strokeWidth fill the column width and give a clear
# "this year was labelled distressed" indicator without a second y-axis.
dist_years = chart_df[chart_df["distressed_next"] == 1]

base = alt.Chart(chart_df).encode(
    x=alt.X(
        "fy_str:O",
        title="Fiscal Year",
        axis=alt.Axis(labelAngle=0),
    ),
)

# Shaded band for distressed years: tall rule spanning full y height.
# strokeWidth=18 ≈ column width for ~15 years; adjust if more years shown.
bands = (
    alt.Chart(dist_years)
    .mark_rule(color="#ffbaba", strokeWidth=18, opacity=0.55)
    .encode(
        x=alt.X("fy_str:O"),
    )
)

# Probability line — single y-axis, domain fixed [0, 1]
prob_line = base.mark_line(color="#c0392b", strokeWidth=2).encode(
    y=alt.Y(
        "lr_prob:Q",
        title="LR predicted P(distressed_next=1)",
        scale=alt.Scale(domain=[0, 1]),
        axis=alt.Axis(format=".2f"),
    ),
    tooltip=[
        alt.Tooltip("fy_str:O", title="FY"),
        alt.Tooltip("lr_prob:Q", title="LR prob", format=".3f"),
        alt.Tooltip("distressed_next:Q", title="distressed_next", format=".0f"),
    ],
)

prob_points = base.mark_point(color="#c0392b", size=60, filled=True).encode(
    y=alt.Y("lr_prob:Q", scale=alt.Scale(domain=[0, 1])),
    tooltip=[
        alt.Tooltip("fy_str:O", title="FY"),
        alt.Tooltip("lr_prob:Q", title="LR prob", format=".3f"),
        alt.Tooltip("distressed_next:Q", title="distressed_next", format=".0f"),
    ],
)

chart = (
    alt.layer(bands, prob_line, prob_points)
    .properties(height=320)
)

st.altair_chart(chart, use_container_width=True)

# Short interpretation note
if n_distressed > 0:
    # Find the first transition year (healthy -> distressed_next=1)
    trans_rows = firm[(firm["distressed"] == 0) & (firm["distressed_next"] == 1)]
    if not trans_rows.empty:
        first_trans_fy = int(trans_rows["fy"].iloc[0])
        lr_at_trans    = trans_rows["lr_prob"].iloc[0]
        if pd.notna(lr_at_trans):
            st.info(
                f"First onset transition: FY {first_trans_fy} "
                f"(healthy → distressed next year).  "
                f"LR probability that year: **{lr_at_trans:.3f}**."
            )
else:
    st.info("No distress events for this firm in the panel.")
