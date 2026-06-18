"""
ingest/edgar_download.py

Downloads XBRL company-facts JSON from the SEC EDGAR API for each ticker
listed in tickers.txt, caching results to data/raw/companyfacts/{TICKER}.json.

SEC fair-use guidelines: descriptive User-Agent, ≤10 req/sec.
"""

import json
import logging
import os
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
TICKERS_FILE = BASE_DIR / "tickers.txt"
CACHE_DIR = BASE_DIR / "data" / "raw" / "companyfacts"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL_TEMPLATE = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

# SEC fair-use policy requires a descriptive User-Agent so they can contact
# you if your script misbehaves.  Format: "Name email@example.com".
# Set the SEC_USER_AGENT environment variable before running:
#   export SEC_USER_AGENT="Your Name your@email.com"    # macOS/Linux
#   set SEC_USER_AGENT=Your Name your@email.com         # Windows CMD
#   $env:SEC_USER_AGENT="Your Name your@email.com"      # Windows PowerShell
USER_AGENT = os.environ.get("SEC_USER_AGENT", "").strip()
if not USER_AGENT:
    raise EnvironmentError(
        "SEC_USER_AGENT environment variable is not set.\n"
        "The SEC requires a descriptive User-Agent for API access.\n"
        "Set it to your name and email, e.g.:\n"
        '  export SEC_USER_AGENT="Jane Smith jane@example.com"'
    )

REQUEST_DELAY = 0.15   # seconds between requests (~6-7 req/sec, well under 10)
BACKOFF_DELAY = 60     # seconds to wait after a 403 or 429

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def fetch_ticker_map(session: requests.Session) -> dict[str, str]:
    """Return {TICKER_UPPER: zero-padded-10-digit-CIK}."""
    log.info("Fetching SEC ticker → CIK map …")
    resp = session.get(TICKER_MAP_URL, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    # The file is a dict of numeric keys → {cik_str, ticker, title}
    mapping: dict[str, str] = {}
    for entry in raw.values():
        ticker = entry["ticker"].upper()
        cik = str(entry["cik_str"]).zfill(10)
        mapping[ticker] = cik
    log.info("Loaded %d tickers from SEC map.", len(mapping))
    return mapping


def load_tickers() -> list[str]:
    """Read tickers.txt, return uppercased, non-empty lines."""
    lines = TICKERS_FILE.read_text().splitlines()
    tickers = [t.strip().upper() for t in lines if t.strip()]
    log.info("Found %d tickers in %s.", len(tickers), TICKERS_FILE)
    return tickers


def download_facts(session: requests.Session, ticker: str, cik: str, dest: Path) -> None:
    """Fetch companyfacts JSON for one ticker and write it to dest."""
    url = FACTS_URL_TEMPLATE.format(cik=cik)
    while True:
        resp = session.get(url, timeout=30)

        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            log.info("SAVED   %s  →  %s  (%d bytes)", ticker, dest.name, len(resp.content))
            return

        if resp.status_code == 404:
            log.warning("SKIP    %s  — 404 no facts found on EDGAR.", ticker)
            return

        if resp.status_code in (403, 429):
            log.warning(
                "BACKOFF %s  — HTTP %d received, waiting %ds …",
                ticker, resp.status_code, BACKOFF_DELAY,
            )
            time.sleep(BACKOFF_DELAY)
            # Retry after backoff
            continue

        # Any other unexpected status: log and move on
        log.error("ERROR   %s  — unexpected HTTP %d, skipping.", ticker, resp.status_code)
        return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    session = make_session()
    ticker_map = fetch_ticker_map(session)
    time.sleep(REQUEST_DELAY)

    tickers = load_tickers()

    skipped_cached = 0
    skipped_unknown = 0
    saved = 0

    for ticker in tickers:
        dest = CACHE_DIR / f"{ticker}.json"

        if dest.exists():
            log.info("CACHED  %s  — already on disk, skipping.", ticker)
            skipped_cached += 1
            continue

        if ticker not in ticker_map:
            log.warning("UNKNOWN %s  — not found in SEC ticker map, skipping.", ticker)
            skipped_unknown += 1
            continue

        cik = ticker_map[ticker]
        download_facts(session, ticker, cik, dest)
        if dest.exists():
            saved += 1

        time.sleep(REQUEST_DELAY)

    log.info(
        "Done. saved=%d  cached=%d  unknown=%d  total=%d",
        saved, skipped_cached, skipped_unknown, len(tickers),
    )


if __name__ == "__main__":
    main()
