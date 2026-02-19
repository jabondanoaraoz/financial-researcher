"""
SEC EDGAR Data Adapter
======================
Official SEC filings data extraction using EDGAR API.

EDGAR provides the most authoritative financial data directly from SEC filings.
All public companies must file quarterly (10-Q) and annual (10-K) reports.

API Documentation: https://www.sec.gov/edgar/sec-api-documentation

Author: Financial Researcher Team
"""

import requests
import json
import time
import re
from typing import Optional, Dict, List, Any
from datetime import datetime
import pandas as pd
import logging
from pathlib import Path

from data.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SEC requires a User-Agent header with contact info
# https://www.sec.gov/os/accessing-edgar-data
HEADERS = {
    'User-Agent': 'Financial-Researcher research@example.com',
    'Accept-Encoding': 'gzip, deflate'
}

# Rate limiting: SEC allows 10 requests/second max
# We use 0.15s delay to be conservative (6.67 req/s)
REQUEST_DELAY = 0.15
_last_request_time = 0


def _rate_limit():
    """Enforce SEC rate limit of 10 requests/second."""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time

    if time_since_last < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - time_since_last
        time.sleep(sleep_time)

    _last_request_time = time.time()


def _make_request(url: str, timeout: int = 30) -> Optional[Dict]:
    """
    Make a request to SEC EDGAR API with proper headers and rate limiting.

    Args:
        url: Full URL to request
        timeout: Request timeout in seconds

    Returns:
        JSON response as dict, or None if request fails
    """
    _rate_limit()

    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"SEC EDGAR: Resource not found at {url}")
        else:
            logger.error(f"SEC EDGAR HTTP error: {e}")
        return None

    except requests.exceptions.RequestException as e:
        logger.error(f"SEC EDGAR request failed: {str(e)}")
        return None

    except json.JSONDecodeError as e:
        logger.error(f"SEC EDGAR: Invalid JSON response: {str(e)}")
        return None


def ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Convert stock ticker to SEC CIK (Central Index Key).

    CIK is the unique identifier used by SEC EDGAR. This function uses
    the official SEC ticker-to-CIK mapping.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        CIK as 10-digit zero-padded string (e.g., '0001652044')
        Returns None if ticker not found
    """
    cache = get_cache()
    ticker_upper = ticker.upper()

    # Check cache first
    cache_key = f"sec_cik_mapping:{ticker_upper}"
    cached_cik = cache.get(cache_key)
    if cached_cik:
        logger.debug(f"CIK cache hit for {ticker_upper}: {cached_cik}")
        return cached_cik

    # Check if we have the full mapping cached
    mapping_cache_key = "sec_ticker_cik_mapping:all"
    ticker_mapping = cache.get(mapping_cache_key)

    if not ticker_mapping:
        # Fetch the official SEC ticker-to-CIK mapping
        logger.info("Fetching SEC ticker-to-CIK mapping...")
        url = "https://www.sec.gov/files/company_tickers.json"

        try:
            _rate_limit()
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Convert to ticker -> CIK mapping
            ticker_mapping = {}
            for entry in data.values():
                ticker_key = entry['ticker'].upper()
                cik_str = str(entry['cik_str']).zfill(10)  # Zero-pad to 10 digits
                ticker_mapping[ticker_key] = cik_str

            # Cache the entire mapping for 7 days (it doesn't change often)
            cache.set(mapping_cache_key, ticker_mapping, ttl_hours=24*7, source="sec_edgar")
            logger.info(f"Cached {len(ticker_mapping)} ticker-to-CIK mappings")

        except Exception as e:
            logger.error(f"Failed to fetch SEC ticker mapping: {str(e)}")
            return None

    # Look up the ticker
    cik = ticker_mapping.get(ticker_upper)

    if cik:
        # Cache individual lookup
        cache.set(cache_key, cik, ttl_hours=24*7, source="sec_edgar")
        logger.info(f"CIK for {ticker_upper}: {cik}")
        return cik
    else:
        logger.warning(f"Ticker {ticker_upper} not found in SEC database")
        return None


def get_xbrl_facts(ticker: str) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Get XBRL financial facts from SEC company facts API.

    Extracts standardized financial metrics from SEC filings (10-K, 10-Q).
    Data is sourced directly from XBRL tags in official filings.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary mapping metric names to DataFrames with columns:
        - period: Reporting period end date
        - value: Metric value
        - filed_date: Date the filing was submitted to SEC
        - form_type: Type of filing (10-K or 10-Q)

        Key metrics included:
        - Revenues, NetIncomeLoss, GrossProfit, OperatingIncome
        - Assets, Liabilities, StockholdersEquity
        - Cash, LongTermDebt, SharesOutstanding, etc.

        Returns None if data extraction fails
    """
    cache = get_cache()
    cache_key = f"sec_xbrl_facts:{ticker.upper()}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"SEC XBRL facts cache hit for {ticker}")
        return cached

    # Get CIK
    cik = ticker_to_cik(ticker)
    if not cik:
        return None

    # Fetch company facts
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    logger.info(f"Fetching SEC XBRL facts for {ticker} (CIK: {cik})")

    data = _make_request(url)
    if not data:
        return None

    # Extract US-GAAP facts (standard accounting taxonomy)
    try:
        us_gaap = data.get('facts', {}).get('us-gaap', {})

        if not us_gaap:
            logger.warning(f"No US-GAAP facts found for {ticker}")
            return None

        # Map common XBRL tags to friendly names
        metric_mapping = {
            'Revenues': 'Revenue',
            'RevenueFromContractWithCustomerExcludingAssessedTax': 'Revenue',
            'NetIncomeLoss': 'NetIncome',
            'GrossProfit': 'GrossProfit',
            'OperatingIncomeLoss': 'OperatingIncome',
            'Assets': 'TotalAssets',
            'Liabilities': 'TotalLiabilities',
            'StockholdersEquity': 'StockholdersEquity',
            'CashAndCashEquivalentsAtCarryingValue': 'CashAndEquivalents',
            'LongTermDebt': 'LongTermDebt',
            'CommonStockSharesOutstanding': 'SharesOutstanding',
            'EarningsPerShareBasic': 'EPS_Basic',
            'EarningsPerShareDiluted': 'EPS_Diluted',
        }

        results = {}

        for xbrl_tag, friendly_name in metric_mapping.items():
            if xbrl_tag not in us_gaap:
                continue

            metric_data = us_gaap[xbrl_tag]

            # Get USD annual and quarterly filings
            records = []

            for unit_type in ['USD', 'shares', 'pure']:
                unit_data = metric_data.get('units', {}).get(unit_type, [])

                for item in unit_data:
                    # Filter to only 10-K and 10-Q
                    form = item.get('form')
                    if form not in ['10-K', '10-Q']:
                        continue

                    # Skip if no end date (we need a period)
                    if 'end' not in item:
                        continue

                    records.append({
                        'period': pd.to_datetime(item['end']),
                        'value': float(item['val']),
                        'filed_date': pd.to_datetime(item['filed']),
                        'form_type': form,
                        'fiscal_year': item.get('fy'),
                        'fiscal_period': item.get('fp')
                    })

            if records:
                # Create DataFrame and sort by period
                df = pd.DataFrame(records)
                df = df.sort_values('period', ascending=False)

                # Remove duplicates (sometimes multiple filings for same period)
                df = df.drop_duplicates(subset=['period', 'form_type'], keep='first')

                results[friendly_name] = df

        if results:
            # Cache for 1 day (new filings come quarterly)
            cache.set(cache_key, results, ttl_hours=24, source="sec_edgar")
            logger.info(f"Extracted {len(results)} metrics from SEC XBRL for {ticker}")
            return results
        else:
            logger.warning(f"No relevant XBRL facts found for {ticker}")
            return None

    except Exception as e:
        logger.error(f"Failed to parse XBRL facts for {ticker}: {str(e)}")
        return None


def get_insider_trades(ticker: str, limit: int = 50) -> Optional[pd.DataFrame]:
    """
    Get insider trading activity from Form 4 filings.

    Form 4 must be filed by company insiders (officers, directors, 10%+ owners)
    within 2 business days of a transaction.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of filings to retrieve (default: 50)

    Returns:
        DataFrame with columns:
        - filing_date: Date Form 4 was filed
        - accession_number: SEC filing identifier
        - filing_url: Link to full filing

        Returns None if data extraction fails

    Note:
        Parsing the actual transaction details from Form 4 XML is complex
        and beyond the scope of this basic implementation. This function
        returns the filing metadata; full parsing would require XML processing.
    """
    cache = get_cache()
    cache_key = f"sec_insider_trades:{ticker.upper()}:{limit}"

    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"SEC insider trades cache hit for {ticker}")
        return cached

    # Get CIK
    cik = ticker_to_cik(ticker)
    if not cik:
        return None

    # Fetch company submissions
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Fetching SEC insider trades for {ticker} (CIK: {cik})")

    data = _make_request(url)
    if not data:
        return None

    try:
        filings = data.get('filings', {}).get('recent', {})

        if not filings:
            logger.warning(f"No recent filings found for {ticker}")
            return None

        # Extract Form 4 filings
        forms = filings.get('form', [])
        filing_dates = filings.get('filingDate', [])
        accession_numbers = filings.get('accessionNumber', [])

        form4_records = []

        for i in range(len(forms)):
            if forms[i] == '4':
                # Clean accession number for URL (remove dashes)
                acc_no = accession_numbers[i]
                acc_no_clean = acc_no.replace('-', '')

                form4_records.append({
                    'filing_date': pd.to_datetime(filing_dates[i]),
                    'accession_number': acc_no,
                    'filing_url': f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={cik}&accession_number={acc_no}&xbrl_type=v"
                })

                if len(form4_records) >= limit:
                    break

        if form4_records:
            df = pd.DataFrame(form4_records)
            df = df.sort_values('filing_date', ascending=False).reset_index(drop=True)

            # Cache for 6 hours (insider trades are time-sensitive)
            cache.set(cache_key, df, ttl_hours=6, source="sec_edgar")
            logger.info(f"Found {len(df)} Form 4 filings for {ticker}")
            return df
        else:
            logger.info(f"No Form 4 filings found for {ticker}")
            return pd.DataFrame()  # Return empty DataFrame instead of None

    except Exception as e:
        logger.error(f"Failed to parse insider trades for {ticker}: {str(e)}")
        return None


def get_filing_text(
    ticker: str,
    filing_type: str = "10-K",
    sections: Optional[List[str]] = None
) -> Optional[Dict[str, str]]:
    """
    Get text content from specific sections of SEC filings.

    Note: This is a simplified implementation that returns filing metadata.
    Full text extraction requires parsing HTML/XML filing documents.

    Args:
        ticker: Stock ticker symbol
        filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
        sections: List of section identifiers (e.g., ['1A'] for Risk Factors)

    Returns:
        Dictionary with:
        - filing_url: URL to the most recent filing
        - filing_date: Date of the filing
        - accession_number: SEC accession number

        Returns None if filing not found

    Note:
        To extract actual section text (e.g., Item 1A - Risk Factors),
        you would need to:
        1. Download the filing HTML/XML
        2. Parse the structure
        3. Extract text between section headers
        This is complex and would require additional libraries like BeautifulSoup.
    """
    if sections is None:
        sections = ['1A']  # Default to Risk Factors

    cache = get_cache()
    cache_key = f"sec_filing:{ticker.upper()}:{filing_type}:{','.join(sections)}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"SEC filing cache hit for {ticker} {filing_type}")
        return cached

    # Get CIK
    cik = ticker_to_cik(ticker)
    if not cik:
        return None

    # Fetch company submissions
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    logger.info(f"Fetching {filing_type} for {ticker} (CIK: {cik})")

    data = _make_request(url)
    if not data:
        return None

    try:
        filings = data.get('filings', {}).get('recent', {})

        if not filings:
            logger.warning(f"No recent filings found for {ticker}")
            return None

        # Find the most recent filing of the requested type
        forms = filings.get('form', [])
        filing_dates = filings.get('filingDate', [])
        accession_numbers = filings.get('accessionNumber', [])
        primary_documents = filings.get('primaryDocument', [])

        for i in range(len(forms)):
            if forms[i] == filing_type:
                acc_no = accession_numbers[i]
                acc_no_clean = acc_no.replace('-', '')
                primary_doc = primary_documents[i]

                # Construct filing URL
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_no_clean}/{primary_doc}"

                result = {
                    'filing_url': filing_url,
                    'filing_date': filing_dates[i],
                    'accession_number': acc_no,
                    'sections_requested': sections,
                    'note': 'Full text extraction requires downloading and parsing the filing document'
                }

                # Cache for 7 days (filings don't change)
                cache.set(cache_key, result, ttl_hours=24*7, source="sec_edgar")
                logger.info(f"Found {filing_type} for {ticker}: {filing_dates[i]}")
                return result

        logger.warning(f"No {filing_type} filings found for {ticker}")
        return None

    except Exception as e:
        logger.error(f"Failed to get {filing_type} for {ticker}: {str(e)}")
        return None


if __name__ == "__main__":
    # Quick test
    print("Testing SEC EDGAR adapter...")

    ticker = "AAPL"

    # Test CIK lookup
    cik = ticker_to_cik(ticker)
    print(f"CIK for {ticker}: {cik}")

    # Test XBRL facts
    facts = get_xbrl_facts(ticker)
    if facts:
        print(f"\nAvailable metrics: {list(facts.keys())}")
        if 'Revenue' in facts:
            print(f"\nRecent Revenue:\n{facts['Revenue'].head()}")
