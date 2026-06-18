"""
features/build_annual.py

Collapses the raw `financials` fact table into one clean row per (cik, fy),
saved as the `annual` table in db/distress.db.

Stock vs Flow — the central design challenge
--------------------------------------------
EDGAR companyfacts packs every historical filing into one JSON blob.  A single
10-K filing contains:

  • Two balance-sheet snapshots     (current year AND prior-year comparative)
  • The full-year income/CF totals  (12-month window)
  • YTD partial-period figures      (3-, 6-, 9-month windows for the same items)

We need exactly ONE number per concept per fiscal year, in the correct window.

  STOCK items (Assets, Liabilities, etc.):
    Instantaneous snapshots — no start date, just an 'end' date.
    Strategy: filter fp='FY', group by (cik, tag, fy), pick earliest filed.

  FLOW items (Revenue, NetIncomeLoss, etc.):
    Duration facts — they have both 'start' and 'end' dates.
    The same filing reports the Q1/Q2/Q3 YTD amounts alongside the full year.
    Strategy: filter fp='FY' AND window length 350–380 days (full year only).
    The window filter is the critical step — without it, a 9-month revenue
    figure shares the same 'end' date and fp='FY' tag as the annual total and
    could be mistakenly selected.

Filing selection rule (both types):
    When multiple filings report the same (cik, tag, fy), keep the EARLIEST
    'filed' date — that is the originally-reported number, before any
    amendment or restatement.  This preserves point-in-time integrity.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "db" / "distress.db"

# ---------------------------------------------------------------------------
# Tag → output-column mapping, classified by type
# ---------------------------------------------------------------------------

# Balance-sheet (stock) items: point-in-time snapshots at fiscal year-end.
# No start date in EDGAR; the 'end' date IS the snapshot date.
STOCK_TAGS: dict[str, str] = {
    "Assets":                             "assets",
    "AssetsCurrent":                      "assets_current",
    "Liabilities":                        "liabilities",
    "LiabilitiesCurrent":                 "liabilities_current",
    "StockholdersEquity":                 "equity",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
}

# Flow items: values that accumulate over a time window.
# Each filing reports multiple windows (Q1/Q2/Q3 YTD + full year).
# Must filter to full-year windows to avoid partial-period contamination.
FLOW_TAGS: dict[str, str] = {
    "Revenues":                                   "revenue",
    "NetIncomeLoss":                              "net_income",
    "OperatingIncomeLoss":                        "operating_income",
    "InterestExpense":                            "interest_expense",
    "NetCashProvidedByUsedInOperatingActivities": "ocf",
}

# Tolerance window for "full year" in days.
# 365 ± 15 covers leap years and fiscal calendar rounding while safely
# excluding 9-month YTD figures (~270 days) and 15-month overlaps.
FULL_YEAR_MIN = 350
FULL_YEAR_MAX = 380

ORDERED_COLS = [
    "cik", "ticker", "fy",
    "assets", "assets_current",
    "liabilities", "liabilities_derived", "liabilities_current",
    "equity", "revenue", "net_income", "operating_income",
    "retained_earnings", "interest_expense", "ocf",
    "filed",
]

# Columns that are actual financial values (excludes the flag column).
# Used for missingness counting and the core-fields check.
VALUE_COLS = [
    "assets", "assets_current", "liabilities", "liabilities_current",
    "equity", "revenue", "net_income", "operating_income",
    "retained_earnings", "interest_expense", "ocf",
]

# Minimum fields required for distress labelling and key ratios.
CORE_COLS = ["assets", "liabilities", "equity", "revenue", "net_income"]

# ---------------------------------------------------------------------------
# Step 1: load raw facts
# ---------------------------------------------------------------------------

def load_financials(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Pull all financials rows joined with form type from filings.
    Parse date columns so window arithmetic (timedelta) works correctly.
    """
    df = pd.read_sql(
        """
        SELECT  f.cik,  f.accn, f.tag,
                f.start, f.end,
                f.fy,   f.fp,
                fi.form, f.filed,
                f.val
        FROM    financials f
        JOIN    filings    fi ON fi.accn = f.accn
        WHERE   f.val IS NOT NULL
        """,
        conn,
        parse_dates=["start", "end", "filed"],
    )

    # Fallback: EDGAR occasionally leaves fy null for older filings.
    # Derive it from the period-end year so no rows are silently orphaned.
    missing_fy = df["fy"].isna()
    if missing_fy.any():
        df.loc[missing_fy, "fy"] = df.loc[missing_fy, "end"].dt.year

    df["fy"] = df["fy"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 2a: select annual value for STOCK (balance-sheet) items
# ---------------------------------------------------------------------------

def select_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each balance-sheet concept, return one row per fiscal year (per cik).

    The core problem (interview talking point):
      EDGAR assigns fp='FY' AND fy=<filing year> to BOTH the current-year AND
      prior-year comparative balance within the SAME 10-K filing.
      Example — FY2017 10-K (filed Nov 2017) contains:
        {fy=2017, end=2017-09-30, val=375B}  ← current year
        {fy=2017, end=2016-09-24, val=321B}  ← prior-year comparative

      Both share fy=2017.  Grouping by (cik, tag, fy) puts both in fy=2017
      and .first() picks whichever pandas returns — often the comparative.
      Grouping by (cik, tag, end) alone still leaves two rows with fy=2009
      after concat (from the two different end dates in the same filing),
      and pivot_table picks the wrong one.

    Fix — two steps:

    Step 1: within each filing (accn), keep ONLY the row with the maximum
      end date.  That is the current-year balance sheet; all earlier end dates
      in the same filing are prior-year comparatives and should be discarded.
      For a FY2017 10-K: max(end) = 2017-09-30 → keep; 2016-09-24 → drop.

    Step 2: among all filings that have the same end date as their current-year
      balance (e.g., the original 10-K and a later 10-K/A amendment), pick
      the EARLIEST filed date.  That is the originally-reported number.
      The fy from that earliest row is always the correct fiscal year because
      it is the current-year filing, not a comparative from a later filing.
    """
    stock = df[
        df["tag"].isin(STOCK_TAGS) &
        (df["fp"] == "FY")
    ].copy()

    # Step 1: identify the current-year balance within each filing.
    # max(end) per (cik, accn, tag) = the balance-sheet date this 10-K is
    # "about"; every other end date in that filing is a comparative.
    max_end = (
        stock.groupby(["cik", "accn", "tag"])["end"]
        .transform("max")
    )
    current_only = stock[stock["end"] == max_end]

    # Step 2: among filings that share the same current-year end date,
    # keep the earliest filed.  fy from that row is the correct fiscal year.
    return (
        current_only
        .sort_values("filed")
        .groupby(["cik", "tag", "end"], as_index=False)
        .first()
    )


# ---------------------------------------------------------------------------
# Step 2b: select annual value for FLOW (duration) items
# ---------------------------------------------------------------------------

def select_flow(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each flow concept, return one row per fiscal year (per cik).

    The same fy=filing-year problem that affects stock items also affects flow
    items.  EDGAR tags ALL rows in a 10-K with the FILING year, not the period
    year.  A FY2012 10-K contains three full-year revenue periods, all fy=2012:
      {fy=2012, start=2009-09-27, end=2010-09-25, val=$65.2B}  ← FY2010 (wrong!)
      {fy=2012, start=2010-09-26, end=2011-09-24, val=$108.2B} ← FY2011 (wrong!)
      {fy=2012, start=2011-09-25, end=2012-09-29, val=$156.5B} ← FY2012 current ✓
    All three pass the full-year window filter (all ~363 days).  Grouping by
    (cik, tag, fy) and calling .first() picks whichever row pandas sees first —
    not necessarily the current-year one.

    Fix (same pattern as select_stock):

    Step A — require a start date (excludes balance-sheet rows automatically).

    Step B — FULL-YEAR WINDOW FILTER.
      Keep only windows between 350 and 380 days.  This discards quarterly
      YTD figures (90/180/270 days) reported alongside the full-year number.

    Step C — CURRENT-YEAR FILTER.
      Within each filing (accn), keep only the flow period whose 'end' date is
      the maximum end date for that (cik, accn, tag).  That is the period the
      10-K is "about" — the current fiscal year.  All earlier end dates in the
      same filing are prior-year comparatives and must be discarded.

    Step D — prefer fp='FY'; accept other fp values only as a fallback when no
      fp='FY' row exists for a (cik, tag, end) after the preceding filters.

    Step E — tie-break: earliest 'filed' date among filings with the same
      current-year end date (e.g., original 10-K vs 10-K/A amendment).
      The fy from that earliest row is always the correct fiscal year.
    """
    # Step A: flow rows only (start date present)
    flow = df[
        df["tag"].isin(FLOW_TAGS) &
        df["start"].notna()
    ].copy()

    # Step B: full-year window filter — discard partial YTD figures
    flow["window_days"] = (flow["end"] - flow["start"]).dt.days
    flow = flow[
        (flow["window_days"] >= FULL_YEAR_MIN) &
        (flow["window_days"] <= FULL_YEAR_MAX)
    ]

    if flow.empty:
        return flow.drop(columns=["window_days"], errors="ignore")

    # Step C: current-year filter.
    # max(end) per (cik, accn, tag) = the period this filing is reporting as
    # current year.  Earlier end dates in the same filing are comparatives.
    max_end = flow.groupby(["cik", "accn", "tag"])["end"].transform("max")
    flow = flow[flow["end"] == max_end]

    # Step D: prefer fp='FY' rows; fall back to non-FY only when absent.
    fy_rows = flow[flow["fp"] == "FY"]
    covered = (
        fy_rows[["cik", "tag", "end"]]
        .drop_duplicates()
        .assign(_covered=True)
    )
    fallback_rows = (
        flow[flow["fp"] != "FY"]
        .merge(covered, on=["cik", "tag", "end"], how="left")
        .query("_covered.isna()")
        .drop(columns="_covered")
    )
    flow_final = pd.concat([fy_rows, fallback_rows], ignore_index=True)

    # Step E: earliest filed per unique current-year end date.
    # Group by (cik, tag, end) — not fy — for the same reason as select_stock:
    # the fy in the earliest-filed row is always the correct fiscal year.
    return (
        flow_final
        .sort_values("filed")
        .groupby(["cik", "tag", "end"], as_index=False)
        .first()
    )


# ---------------------------------------------------------------------------
# Step 3: pivot long → wide and attach metadata
# ---------------------------------------------------------------------------

def build_annual(conn: sqlite3.Connection) -> pd.DataFrame:
    df    = load_financials(conn)
    stock = select_stock(df)
    flow  = select_flow(df)

    # Combine; STOCK_TAGS and FLOW_TAGS are disjoint so no key collisions
    long = pd.concat([stock, flow], ignore_index=True)

    # Map EDGAR tag names to short column names for the output table
    all_tags = {**STOCK_TAGS, **FLOW_TAGS}
    long["col"] = long["tag"].map(all_tags)

    # 'filed' anchor for the annual row: use the Assets filing date.
    # Assets is present in virtually every 10-K, making it a reliable proxy
    # for "when was this fiscal year first reported publicly."
    filed_anchor = (
        long[long["tag"] == "Assets"][["cik", "fy", "filed"]]
        .sort_values("filed")
        .groupby(["cik", "fy"], as_index=False)
        .first()
    )

    # Pivot long → wide.  Each (cik, fy, col) is already unique after the
    # dedup in select_stock / select_flow, so aggfunc="first" is a formality.
    pivot = (
        long
        .pivot_table(
            index=["cik", "fy"],
            columns="col",
            values="val",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns.name = None

    # Ensure all expected columns exist (a tag could be absent for every firm)
    for col in all_tags.values():
        if col not in pivot.columns:
            pivot[col] = float("nan")

    # Attach filed anchor and ticker
    pivot = pivot.merge(filed_anchor, on=["cik", "fy"], how="left")
    companies = pd.read_sql("SELECT cik, ticker FROM companies", conn)
    pivot = pivot.merge(companies, on="cik", how="left")

    # -- Derive liabilities = assets − equity where the tag is missing -------
    #
    # Some firms (e.g. Macy's) do not tag total liabilities in EDGAR; their
    # XBRL only contains LiabilitiesAndStockholdersEquity (= total assets).
    # The balance-sheet identity assets = liabilities + equity lets us recover
    # liabilities exactly.  We flag derived rows so downstream analysis can
    # treat them differently if needed.
    #
    # We do NOT derive when either assets or equity is null — fabricating a
    # chain of inferences would introduce undetectable errors.
    pivot["liabilities_derived"] = False
    derive_mask = (
        pivot["liabilities"].isna()
        & pivot["assets"].notna()
        & pivot["equity"].notna()
    )
    pivot.loc[derive_mask, "liabilities"]         = pivot.loc[derive_mask, "assets"] - pivot.loc[derive_mask, "equity"]
    pivot.loc[derive_mask, "liabilities_derived"] = True

    # Sanity check: derived liabilities should satisfy 0 ≤ liabilities ≤ assets.
    # A negative value means equity > assets (possible but rare; signals deep
    # negative retained earnings).  Liabilities > assets means equity is
    # negative — print a warning but do NOT drop the row.
    if derive_mask.any():
        bad_neg = pivot[derive_mask & (pivot["liabilities"] < 0)]
        bad_big = pivot[derive_mask & (pivot["liabilities"] > pivot["assets"])]
        if not bad_neg.empty:
            print(f"  [WARN] {len(bad_neg)} derived-liabilities row(s) are negative "
                  f"(equity > assets):")
            print(bad_neg[["ticker", "fy", "assets", "equity", "liabilities"]].to_string(index=False))
        if not bad_big.empty:
            print(f"  [WARN] {len(bad_big)} derived-liabilities row(s) exceed assets "
                  f"(negative equity):")
            print(bad_big[["ticker", "fy", "assets", "equity", "liabilities"]].to_string(index=False))

    final_cols = [c for c in ORDERED_COLS if c in pivot.columns]
    return (
        pivot[final_cols]
        .sort_values(["ticker", "fy"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_aapl(annual: pd.DataFrame) -> None:
    aapl = (
        annual[annual["ticker"] == "AAPL"]
        .set_index("fy")
        [["assets", "revenue", "net_income", "operating_income", "equity", "filed"]]
    )
    print("=" * 100)
    print("AAPL — all fiscal years (USD, raw)")
    print("=" * 100)
    with pd.option_context(
        "display.float_format", lambda x: f"{x:>20,.0f}" if pd.notna(x) else f"{'NaN':>20}",
        "display.max_columns", 10,
        "display.width", 120,
    ):
        print(aapl.to_string())
    print()


def print_completeness(annual: pd.DataFrame) -> None:
    val_cols = [c for c in ORDERED_COLS if c not in ("cik", "ticker", "fy", "filed")]
    annual = annual.copy()
    annual["_complete"] = annual[val_cols].notna().all(axis=1)
    summary = (
        annual
        .groupby("ticker", as_index=False)
        .agg(
            total_years    = ("fy",        "count"),
            complete_years = ("_complete", "sum"),
        )
        .sort_values("ticker")
    )
    print("=" * 50)
    print("Complete rows per firm (all 11 fields non-null)")
    print("=" * 50)
    print(summary.to_string(index=False))
    print()


def print_missingness_grid(annual: pd.DataFrame) -> None:
    """10 × 11 grid of non-null counts per firm per financial field."""
    grid = (
        annual.groupby("ticker")[VALUE_COLS]
        .apply(lambda g: g.notna().sum())
        .astype(int)
    )
    grid.insert(0, "total_yrs", annual.groupby("ticker")["fy"].count())
    # flag derived-liabilities firms
    derived_tickers = set(annual.loc[annual["liabilities_derived"] == True, "ticker"])
    if derived_tickers:
        grid.index = [
            f"{t}*" if t in derived_tickers else t for t in grid.index
        ]
    print("=" * 135)
    print("MISSINGNESS GRID  (non-null count per firm per field; * = liabilities derived via assets-equity)")
    print("=" * 135)
    with pd.option_context("display.width", 160, "display.max_columns", 15):
        print(grid.to_string())
    print()


def print_completeness(annual: pd.DataFrame) -> None:
    """Complete (all-11-fields non-null) rows per firm, plus core-fields coverage."""
    ann = annual.copy()
    ann["_all_complete"]  = ann[VALUE_COLS].notna().all(axis=1)
    ann["_core_complete"] = ann[CORE_COLS].notna().all(axis=1)

    summary = (
        ann.groupby("ticker", as_index=False)
        .agg(
            total_yrs        = ("fy",             "count"),
            all_11_complete  = ("_all_complete",  "sum"),
            core_5_complete  = ("_core_complete", "sum"),
        )
        .sort_values("ticker")
    )
    print("=" * 60)
    print("COMPLETENESS  (all-11 = every field non-null; core-5 = assets/liabilities/equity/revenue/net_income)")
    print("=" * 60)
    print(summary.to_string(index=False))
    print()

    total_core = ann["_core_complete"].sum()
    print(f"  Working sample (core-5 complete): {total_core} firm-year rows")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn   = sqlite3.connect(DB_PATH)
    annual = build_annual(conn)

    annual.to_sql("annual", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(annual)} rows to 'annual' table in {DB_PATH}\n")

    print_aapl(annual)
    print_missingness_grid(annual)
    print_completeness(annual)


if __name__ == "__main__":
    main()
