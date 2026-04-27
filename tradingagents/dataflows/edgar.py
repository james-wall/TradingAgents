"""SEC EDGAR data fetcher.

EDGAR is free and open. Required headers:
- User-Agent identifying the requester (set EDGAR_USER_AGENT env var)
- Rate limit: 10 req/sec (we don't enforce, just don't fan out)

Filings fetched here are immutable, so we cache everything to disk by
accession number. The ticker→CIK mapping changes rarely; cached for 1 day.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

DEFAULT_USER_AGENT = "TradingAgents Research research@example.com"
USER_AGENT = os.getenv("EDGAR_USER_AGENT", DEFAULT_USER_AGENT)
CACHE_DIR = Path(os.getenv("EDGAR_CACHE_DIR", str(Path.home() / ".cache" / "tradingagents" / "edgar")))

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_clean}/{filename}"


def _headers() -> dict:
    return {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _cache_dir() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def ticker_to_cik(ticker: str) -> Optional[str]:
    """Map a ticker to its 10-digit zero-padded SEC CIK string. None if unknown."""
    cache_file = _cache_dir() / "ticker_cik_map.json"
    fresh = cache_file.exists() and (
        datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
    ) < timedelta(days=1)

    if fresh:
        data = json.loads(cache_file.read_text())
    else:
        r = requests.get(TICKER_CIK_URL, headers=_headers(), timeout=30)
        r.raise_for_status()
        data = r.json()
        cache_file.write_text(json.dumps(data))

    target = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == target:
            return f"{int(entry['cik_str']):010d}"
    return None


def get_recent_filings(
    ticker: str,
    filing_types: Optional[list[str]] = None,
    days_back: int = 120,
) -> list[dict]:
    """Return recent filings for a ticker, newest first.

    Each entry: {accession_no, filing_type, filed_date (YYYY-MM-DD), url}.
    """
    if filing_types is None:
        filing_types = ["10-K", "10-Q"]

    cik = ticker_to_cik(ticker)
    if not cik:
        return []

    r = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=_headers(), timeout=30)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accs = recent.get("accessionNumber", [])
    primaries = recent.get("primaryDocument", [])

    cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    cik_int = int(cik)

    results = []
    for form, fdate, acc, prim in zip(forms, dates, accs, primaries):
        if form not in filing_types:
            continue
        if fdate < cutoff:
            continue
        results.append({
            "ticker": ticker.upper(),
            "accession_no": acc,
            "filing_type": form,
            "filed_date": fdate,
            "url": ARCHIVE_URL.format(
                cik_int=cik_int,
                accession_clean=acc.replace("-", ""),
                filename=prim,
            ),
        })

    # SEC returns newest first already, but be explicit
    results.sort(key=lambda x: x["filed_date"], reverse=True)
    return results


def get_filing_text(filing: dict, max_chars: int = 200_000) -> str:
    """Fetch the primary document text for a filing. Cached by accession number.

    Strips HTML, collapses whitespace. Truncates to max_chars (filings can be huge
    — 10-Ks routinely run 500k+ chars and most LLMs can't handle that in one call).
    """
    cache_file = _cache_dir() / f"{filing['accession_no'].replace('-', '')}.txt"

    if cache_file.exists():
        return cache_file.read_text()[:max_chars]

    # Be polite to EDGAR (10 req/sec is the documented cap)
    time.sleep(0.15)

    r = requests.get(filing["url"], headers=_headers(), timeout=60)
    r.raise_for_status()

    if filing["url"].lower().endswith((".htm", ".html")):
        # Lazy import — bs4 is heavyweight and only needed for HTML filings
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    else:
        text = r.text

    # Collapse runs of blank lines and multiple spaces
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    cache_file.write_text(text)
    return text[:max_chars]
