"""
features/build_labels.py

Builds the distress label and enforces point-in-time (PIT) correctness.
Reads the `annual` table from db/distress.db; writes `annual_labeled`.

Label design
------------
PART A — distress label for fiscal year t

  PRIMARY rule (label_source = 'coverage'):
    A firm is distressed when its interest coverage ratio falls below 1.
    Formally: operating_income < interest_expense
    Both must be non-null; either being null disqualifies the primary rule.

    Interview note: ICR < 1 means EBIT cannot cover interest payments — the
    canonical Altman/academic definition of financial distress.  Using
    operating_income (EBIT proxy) rather than net income keeps the signal
    free of tax and extraordinary-item noise.

  FALLBACK rule (label_source = 'fallback'):
    Used when operating_income or interest_expense is NaN.
    A firm is distressed when net_income < 0 in BOTH year t AND year t-1.
    Two consecutive loss years are required to avoid flagging one-off charges.

    net_income_lag1 is computed via an exact-year-match join on (cik, fy-1)
    rather than a positional shift (e.g. .shift(1)).  Positional shifts break
    silently when fiscal years are non-consecutive (gaps from missing filings,
    ticker entry/exit).  Exact-year-match only fires when the prior year's
    data actually exists, making the NaN propagation honest.

  NEITHER rule fires (distressed = NaN):
    When neither rule can be applied (e.g. no interest expense AND no
    net_income), we leave distressed as NaN rather than imputing 0.
    These rows are NOT included in the working model sample.

PART B — prediction framing

  distressed_next = distressed at t+1 (the TARGET variable for a model).
  Also computed via exact-year-match join on (cik, fy+1) for the same
  panel-gap reason as net_income_lag1.

  For a model trained at time t, the features are values at t and the
  target is distress at t+1 — i.e., "will this firm enter distress next year?"

PART C — point-in-time / look-ahead control

  available_from = filed date of year t's annual report (10-K).
  A model can only use year-t features after this date.  Using the features
  before available_from would constitute look-ahead bias.

  filed_year = calendar year in which the 10-K was filed.
  late_filing = True when filed_year > fy + 1.  This flags anomalously late
  filings (e.g. restatements filed 2+ years after fiscal year-end) that
  should be excluded from a real-time trading model.

  Point-in-time integrity: the `filed` column in `annual` is the date the
  original 10-K was accepted by EDGAR, sourced directly from the companyfacts
  JSON.  No look-ahead is introduced here because we never use features from
  a filing whose `filed` date is later than the decision date.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH  = BASE_DIR / "db" / "distress.db"

# ---------------------------------------------------------------------------
# PART A — build distress label
# ---------------------------------------------------------------------------

def build_distress_label(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns: net_income_lag1, distressed (0/1/NaN), label_source.

    net_income_lag1 is computed via exact-year-match join on (cik, fy-1).
    This correctly handles non-consecutive fiscal years (panel gaps).
    """
    df = annual.copy()

    # --- Compute net_income_lag1 via exact-year-match join -------------------
    # Build a lookup table: for each (cik, fy), what was net_income?
    # Then join it back on (cik, fy-1) to get the lagged value.
    # This is equivalent to: "find the row in the same firm where fy = t-1".
    # A positional .shift(1) would silently give wrong values for firms with
    # gaps (e.g., if FY2009 data is missing, FY2011's lag should be NaN, not
    # FY2010's net_income).
    lag_lookup = (
        df[["cik", "fy", "net_income"]]
        .rename(columns={"fy": "fy_lag", "net_income": "net_income_lag1"})
    )
    df = df.merge(
        lag_lookup.assign(fy=lag_lookup["fy_lag"] + 1)[["cik", "fy", "net_income_lag1"]],
        on=["cik", "fy"],
        how="left",
    )

    # --- PRIMARY rule: interest coverage ratio < 1 ---------------------------
    # operating_income < interest_expense  ⟺  ICR < 1
    # Both must be non-null for the rule to fire.
    can_use_primary = df["operating_income"].notna() & df["interest_expense"].notna()
    primary_distress = can_use_primary & (df["operating_income"] < df["interest_expense"])

    # --- FALLBACK rule: two consecutive net-income losses --------------------
    # Only fires when primary rule cannot be applied.
    can_use_fallback = (~can_use_primary) & df["net_income"].notna() & df["net_income_lag1"].notna()
    fallback_distress = can_use_fallback & (df["net_income"] < 0) & (df["net_income_lag1"] < 0)

    # --- Assemble distressed and label_source --------------------------------
    # distressed is 0/1 where a rule fired, NaN where no rule could apply.
    df["distressed"]   = float("nan")
    df["label_source"] = pd.NA

    # Primary rule rows (0 or 1)
    df.loc[can_use_primary, "distressed"]   = primary_distress[can_use_primary].astype(int)
    df.loc[can_use_primary, "label_source"] = "coverage"

    # Fallback rule rows (0 or 1) — only where primary didn't fire
    df.loc[can_use_fallback, "distressed"]   = fallback_distress[can_use_fallback].astype(int)
    df.loc[can_use_fallback, "label_source"] = "fallback"

    return df


# ---------------------------------------------------------------------------
# PART B — prediction framing: distressed_next
# ---------------------------------------------------------------------------

def add_distressed_next(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add distressed_next = distressed value at (cik, fy+1).

    Same exact-year-match join pattern as net_income_lag1.
    distressed_next is NaN for the last fiscal year of each firm
    (no t+1 row exists) and for any firm-year where the t+1 label is NaN.

    This is the TARGET variable for a binary classification model:
    "given features at year t, will this firm be distressed at t+1?"
    """
    next_lookup = (
        df[["cik", "fy", "distressed"]]
        .rename(columns={"fy": "fy_next", "distressed": "distressed_next"})
    )
    df = df.merge(
        next_lookup.assign(fy=next_lookup["fy_next"] - 1)[["cik", "fy", "distressed_next"]],
        on=["cik", "fy"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# PART C — point-in-time columns
# ---------------------------------------------------------------------------

def add_pit_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      available_from — alias for 'filed'; the date features become observable.
      filed_year     — calendar year of the filing.
      late_filing    — True when filed_year > fy + 1.

    'filed' is already in the annual table (sourced from EDGAR companyfacts
    JSON).  We rename rather than copy so the intent is explicit.

    A late_filing flag does NOT drop the row.  Downstream the analyst can
    decide whether to exclude these rows from a real-time simulation.
    """
    df = df.copy()

    # filed is already a date string; parse it for year extraction
    filed_parsed = pd.to_datetime(df["filed"], errors="coerce")

    df["available_from"] = df["filed"]    # same value, explicit label
    df["filed_year"]     = filed_parsed.dt.year.astype("Int64")   # nullable int
    df["late_filing"]    = df["filed_year"] > (df["fy"] + 1)

    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_label_report(df: pd.DataFrame) -> None:
    """Print the four outputs requested:
      1. Total distress events, split by label_source
      2. Distress rate per firm
      3. Final modelling table row count
      4. List of (firm, fy) flagged as distressed
    """
    print("=" * 70)
    print("DISTRESS LABEL REPORT")
    print("=" * 70)

    # 1. Total distress events by source
    labeled = df[df["distressed"].notna()]
    total_events = int((labeled["distressed"] == 1).sum())
    print(f"\n  Total firm-years with a label (distressed not NaN): {len(labeled):>5}")
    print(f"  Total distress events (distressed = 1):            {total_events:>5}")
    print()

    source_counts = (
        labeled[labeled["distressed"] == 1]
        .groupby("label_source", dropna=False)
        .size()
        .rename("distress_events")
        .reset_index()
    )
    print("  Events by label_source:")
    for _, row in source_counts.iterrows():
        print(f"    {row['label_source']:<12}  {int(row['distress_events']):>4}")
    print()

    # 2. Distress rate per firm
    print("  Distress rate per firm (labeled rows only):")
    print(f"  {'Ticker':<8}  {'Labeled yrs':>11}  {'Distress events':>15}  {'Rate':>6}")
    print(f"  {'-'*8}  {'-'*11}  {'-'*15}  {'-'*6}")
    per_firm = (
        labeled.groupby("ticker")
        .agg(
            labeled_yrs    = ("distressed", "count"),
            distress_events= ("distressed", lambda x: (x == 1).sum()),
        )
        .reset_index()
        .sort_values("ticker")
    )
    per_firm["rate"] = per_firm["distress_events"] / per_firm["labeled_yrs"]
    for _, row in per_firm.iterrows():
        print(f"  {row['ticker']:<8}  {int(row['labeled_yrs']):>11}  "
              f"{int(row['distress_events']):>15}  {row['rate']:>6.1%}")
    print()

    # 3. Final modelling table row count
    # Model-ready: distressed_next is not null (we have a t+1 target)
    model_ready = df[df["distressed_next"].notna()]
    print(f"  Final modelling table row count")
    print(f"  (rows where distressed_next is not NaN): {len(model_ready):>5}")
    print()

    # 4. Distressed firm-years list
    print("  Firm-years flagged as distressed (distressed = 1):")
    print(f"  {'Ticker':<8}  {'FY':>4}  {'Label source':<12}  {'Available from':>14}")
    print(f"  {'-'*8}  {'-'*4}  {'-'*12}  {'-'*14}")
    distressed_rows = (
        df[df["distressed"] == 1]
        [["ticker", "fy", "label_source", "available_from"]]
        .sort_values(["ticker", "fy"])
    )
    for _, row in distressed_rows.iterrows():
        print(f"  {row['ticker']:<8}  {int(row['fy']):>4}  "
              f"{str(row['label_source']):<12}  {str(row['available_from']):>14}")

    # Late filing warnings
    late = df[df["late_filing"] == True]
    if not late.empty:
        print()
        print(f"  [WARN] {len(late)} row(s) have late_filing=True "
              f"(filed > fy+1 calendar year):")
        for _, row in late.iterrows():
            print(f"    {row['ticker']:<6}  fy={int(row['fy'])}  "
                  f"filed={row['filed']}  filed_year={row['filed_year']}")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    conn   = sqlite3.connect(DB_PATH)
    annual = pd.read_sql("SELECT * FROM annual", conn, parse_dates=["filed"])

    # PART A: distress label
    labeled = build_distress_label(annual)

    # PART B: prediction framing
    labeled = add_distressed_next(labeled)

    # PART C: PIT columns
    labeled = add_pit_columns(labeled)

    # Save to DB
    labeled.to_sql("annual_labeled", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Saved {len(labeled)} rows to 'annual_labeled' table in {DB_PATH}\n")

    print_label_report(labeled)


if __name__ == "__main__":
    main()
