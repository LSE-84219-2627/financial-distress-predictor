"""
db/build_db.py

Parses cached SEC EDGAR companyfacts JSON files and loads them into a local
SQLite database (db/distress.db).

Schema design (interview-ready summary)
---------------------------------------
companies  — one row per firm; the authoritative CIK is the natural key because
             tickers can be reassigned after delistings.

filings    — one row per SEC submission (accession number = accn).  Normalised
             out of financials so that filing metadata (form type, period) is
             stored once rather than repeated on every financial row.

financials — the fact table; one row per (company × filing × tag × period).
             The natural key is (cik, accn, tag, start, end) because a single
             filing reports comparative periods: a 10-K contains both the
             current and prior year-end balance for Assets, and a 10-Q reports
             YTD flow figures with different start dates but the same end date.
             Using (cik, accn, tag) alone silently drops ~50% of facts via
             INSERT OR REPLACE — confirmed: AAPL Assets had 144 raw entries
             but only 71 survived the old key.  The fix adds `start` (NULL for
             instantaneous/balance-sheet facts) and includes it in the PK,
             treating NULL as '' for uniqueness purposes.

Re-runnable: tables are dropped and recreated on every run, so the script is
idempotent — safe to re-run after downloading new tickers.
"""

import json
import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
CACHE_DIR  = BASE_DIR / "data" / "raw" / "companyfacts"
DB_PATH    = BASE_DIR / "db" / "distress.db"

# ---------------------------------------------------------------------------
# Tags we care about
#
# Each entry is the canonical tag name we store in the DB.  Where EDGAR uses
# multiple tag names for the same concept (e.g. revenue), we unify them here.
# ---------------------------------------------------------------------------

PRIMARY_TAGS = [
    "Assets",
    "AssetsCurrent",
    "Liabilities",
    "LiabilitiesCurrent",
    "StockholdersEquity",
    "Revenues",
    "NetIncomeLoss",
    "OperatingIncomeLoss",
    "RetainedEarningsAccumulatedDeficit",
    "InterestExpense",
    "NetCashProvidedByUsedInOperatingActivities",
]

# Tag alias map  {canonical DB tag → [source EDGAR tags, in priority order]}
#
# For every canonical tag, aliases are tried in order.  Deduplication by
# (accn, start, end) means the FIRST alias that covers a period wins; later
# aliases only fill gaps.  This implements two distinct aliasing strategies:
#
#   COMBINE-ALL  (revenue): multiple tags cover non-overlapping date ranges,
#     so we union all of them.  Since the same period is rarely filed under
#     two tags simultaneously, dedup rarely fires here.
#
#   PREFER-BASE  (equity, ocf): the base tag is the conceptually correct one;
#     the fallback fills years where only the more verbose tag was filed.
#     Processing base first means the fallback NEVER overwrites a base value.
#
# Interview point: this is equivalent to a COALESCE across tag names, applied
# before the data enters the DB, so downstream queries stay simple.
TAG_ALIASES: dict[str, list[str]] = {
    # -- Revenue: unite three eras of GAAP tagging ----------------------------
    # Pre-ASC-606 filers used SalesRevenueNet (goods) or SalesRevenueGoodsNet
    # (goods-only retailers like GME).  The 2018 standard introduced the long
    # RevenueFromContract... tag.  All three cover non-overlapping periods for
    # most companies, so we union them all.
    "Revenues": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",     # goods-only retailer tag (e.g. GME pre-2018)
    ],

    # -- Equity: base tag for simple equity; fallback includes NCI ------------
    # GME and other firms with noncontrolling interests (subsidiaries) filed
    # under the longer tag for most of their history.  We want the simpler
    # StockholdersEquity when available, because it matches the NCI-excluded
    # definition used by distress models.  The fallback fills the gaps where
    # only the NCI-inclusive tag was filed.
    "StockholdersEquity": [
        "StockholdersEquity",       # base: common equity only
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],

    # -- OCF: total preferred; continuing-operations is the fallback ----------
    # Ford and some conglomerates primarily file the ContinuingOperations
    # variant.  The total tag is preferred when present (e.g. post-restructure
    # years) so we don't accidentally drop discontinued-operations cash flows.
    "NetCashProvidedByUsedInOperatingActivities": [
        "NetCashProvidedByUsedInOperatingActivities",    # base: total OCF
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL = """
-- Drop tables so re-runs start clean (idempotent).
DROP TABLE IF EXISTS financials;
DROP TABLE IF EXISTS filings;
DROP TABLE IF EXISTS companies;

-- companies: one row per firm.
-- CIK is the SEC's permanent identifier; ticker is stored for convenience
-- but is NOT the primary key because tickers get recycled after delistings.
CREATE TABLE companies (
    cik    INTEGER PRIMARY KEY,
    ticker TEXT    NOT NULL,
    name   TEXT    NOT NULL
);

-- filings: one row per SEC submission.
-- accn (accession number) is globally unique in EDGAR.
-- Normalising filing metadata here avoids repeating form/fy/fp on every
-- financial fact row — standard 3NF.
CREATE TABLE filings (
    accn   TEXT    PRIMARY KEY,
    cik    INTEGER NOT NULL REFERENCES companies(cik),
    form   TEXT,           -- e.g. "10-K", "10-Q"
    fy     INTEGER,        -- fiscal year
    fp     TEXT,           -- fiscal period: "FY", "Q1"–"Q4"
    filed  DATE            -- date SEC received the filing
);

-- financials: the fact table.
--
-- Natural key fix (interview talking point):
--   The old key (cik, accn, tag) assumed one period per filing per concept,
--   but EDGAR includes comparative periods in each filing.  A 10-K reports
--   Assets at both the current and prior fiscal year-end; a 10-Q reports
--   YTD cash flows for the 3-month and 9-month periods, all with the same
--   'end' date but different 'start' dates.  Adding 'start' to the key
--   preserves every observation.
--
-- 'start' is NULL for instantaneous facts (balance-sheet items like Assets)
--   and a date for duration facts (flow items like NetIncomeLoss).
-- 'end'   is always present — the period-end or balance-sheet date.
-- 'val'   is in USD (raw, not thousands) as reported by EDGAR.
--
-- SQLite does not allow expressions in a PRIMARY KEY, so we use a surrogate
-- column 'start_key': the Python loader sets it to the start date string when
-- present, or '' for balance-sheet facts that have no start date.  This lets
-- the composite PK enforce uniqueness across all observation windows without
-- storing NULLs in a key column (which SQLite treats as always distinct).
CREATE TABLE financials (
    cik       INTEGER NOT NULL REFERENCES companies(cik),
    accn      TEXT    NOT NULL REFERENCES filings(accn),
    tag       TEXT    NOT NULL,   -- canonical tag name (see PRIMARY_TAGS)
    start_key TEXT    NOT NULL,   -- start date string, or '' for snapshots
    start     DATE,               -- period start; NULL for balance-sheet snapshots
    end       DATE    NOT NULL,   -- period end or balance-sheet date
    fy        INTEGER,
    fp        TEXT,
    filed     DATE,
    val       REAL,
    PRIMARY KEY (cik, accn, tag, start_key, end)
);
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_usd_facts(facts_blob: dict, tag: str) -> list[dict]:
    """Return the list of USD fact records for a given us-gaap tag, or []."""
    try:
        return facts_blob["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError:
        return []


def resolve_tags(facts_blob: dict, canonical_tag: str) -> list[dict]:
    """
    Resolve a canonical DB tag to its EDGAR source records.

    Iterates the alias list for `canonical_tag` (falling back to a single-item
    list of the tag itself when no aliases are defined).  Deduplicates by
    (accn, start, end): the FIRST alias that covers a period wins, so the base
    tag always takes priority over fallback tags.

    This implements both strategies in TAG_ALIASES:
      - COMBINE-ALL  (revenue): union across all aliases; dedup rarely fires.
      - PREFER-BASE  (equity, ocf): base processed first → fallback only fills
        genuinely missing periods, never overwrites a base-tag value.
    """
    aliases = TAG_ALIASES.get(canonical_tag, [canonical_tag])
    seen: set[tuple] = set()
    combined: list[dict] = []
    for source_tag in aliases:
        for rec in get_usd_facts(facts_blob, source_tag):
            key = (rec.get("accn"), rec.get("start"), rec.get("end"))
            if key not in seen:
                seen.add(key)
                combined.append(rec)
    return combined

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # Create schema (drops first for idempotency)
    cur.executescript(DDL)
    conn.commit()
    print(f"Schema created in {DB_PATH}\n")

    json_files = sorted(CACHE_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {CACHE_DIR}. Run ingest/edgar_download.py first.")
        return

    # Track unique accn values across all companies to avoid FK violations
    # when the same filing appears in multiple tag lists (it won't, but be safe).
    seen_accns: set[str] = set()

    for json_path in json_files:
        ticker = json_path.stem                         # filename without .json
        blob   = json.loads(json_path.read_bytes())

        cik  = blob["cik"]
        name = blob.get("entityName", ticker)

        # -- companies -------------------------------------------------------
        cur.execute(
            "INSERT OR REPLACE INTO companies (cik, ticker, name) VALUES (?, ?, ?)",
            (cik, ticker, name),
        )
        print(f"  {ticker:6s}  CIK={cik}  {name}")

        # -- walk each tag ---------------------------------------------------
        for tag in PRIMARY_TAGS:
            # resolve_tags handles both the single-tag case (no aliases defined)
            # and the multi-alias case (revenue, equity, ocf) transparently.
            records = resolve_tags(blob, tag)

            for rec in records:
                accn  = rec.get("accn")
                filed = rec.get("filed")
                fy    = rec.get("fy")
                fp    = rec.get("fp")
                form  = rec.get("form")
                # 'start' is present on duration/flow facts (e.g. revenues for
                # a quarter), absent on instantaneous facts (e.g. total assets
                # at a balance-sheet date).  Store None so it lands as NULL.
                start = rec.get("start")        # may be None — that's correct
                end   = rec.get("end")
                val   = rec.get("val")

                # Skip records with no accession number (can't link to filing)
                if not accn or not end:
                    continue

                # -- filings (insert once per accn) --------------------------
                if accn not in seen_accns:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO filings (accn, cik, form, fy, fp, filed)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (accn, cik, form, fy, fp, filed),
                    )
                    seen_accns.add(accn)

                # -- financials ----------------------------------------------
                # start_key is '' for balance-sheet snapshots (no start date)
                # and the ISO date string for duration/flow facts.  It sits in
                # the PK so every observation window within a filing gets its
                # own row — no comparative periods are silently overwritten.
                start_key = start if start is not None else ""
                cur.execute(
                    """
                    INSERT OR REPLACE INTO financials
                        (cik, accn, tag, start_key, start, end, fy, fp, filed, val)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (cik, accn, tag, start_key, start, end, fy, fp, filed, val),
                )

        conn.commit()   # commit after each company so progress isn't lost

    # -- Summary -------------------------------------------------------------
    print("\n" + "=" * 55)
    print("DATABASE SUMMARY")
    print("=" * 55)

    n_companies = cur.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    n_filings   = cur.execute("SELECT COUNT(*) FROM filings").fetchone()[0]
    n_facts     = cur.execute("SELECT COUNT(*) FROM financials").fetchone()[0]

    print(f"  Companies : {n_companies:>7,}")
    print(f"  Filings   : {n_filings:>7,}")
    print(f"  Facts     : {n_facts:>7,}")
    print()
    print(f"  {'Tag':<50}  {'Rows':>7}")
    print(f"  {'-'*50}  {'-'*7}")

    rows = cur.execute(
        """
        SELECT tag, COUNT(*) AS n
        FROM   financials
        GROUP  BY tag
        ORDER  BY n DESC
        """
    ).fetchall()
    for tag, n in rows:
        print(f"  {tag:<50}  {n:>7,}")

    # -- Annual observations per firm ----------------------------------------
    # For modelling we want annual snapshots (fp='FY'), not quarterly noise.
    # This query counts how many distinct fiscal years each company has for
    # 'Assets' with fp='FY' — a proxy for clean annual panel coverage.
    print()
    print(f"  {'Annual FY observations per company (tag=Assets, fp=FY)'}")
    print(f"  {'-'*55}")
    annual = cur.execute(
        """
        SELECT   c.ticker, c.name, COUNT(DISTINCT f.fy) AS fy_count
        FROM     financials f
        JOIN     companies  c ON c.cik = f.cik
        WHERE    f.tag = 'Assets'
          AND    f.fp  = 'FY'
        GROUP BY f.cik
        ORDER BY fy_count DESC
        """
    ).fetchall()
    for ticker, name, fy_count in annual:
        print(f"  {ticker:6s}  {name:<40}  {fy_count:>3} years")

    print("=" * 55)
    conn.close()


if __name__ == "__main__":
    main()
