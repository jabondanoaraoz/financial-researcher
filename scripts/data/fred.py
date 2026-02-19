"""
FRED Data Adapter
=================
Federal Reserve Economic Data (FRED) - Macroeconomic indicators.

FRED provides authoritative US economic data from the Federal Reserve Bank
of St. Louis. Essential for understanding the macro environment for investments.

API Docs: https://fred.stlouisfed.org/docs/api/

Author: Financial Researcher Team
"""

import requests
import os
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import pandas as pd

from data.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"


def _get_api_key() -> Optional[str]:
    """Get FRED API key from environment."""
    api_key = os.getenv("FRED_API_KEY")

    if not api_key or api_key == "your_fred_api_key_here":
        logger.warning("FRED API key not configured - using default values")
        return None

    return api_key


def _make_request(endpoint: str, params: Dict[str, str]) -> Optional[Dict]:
    """
    Make a request to FRED API.

    Args:
        endpoint: API endpoint (e.g., 'series/observations')
        params: Query parameters

    Returns:
        JSON response as dict, or None if request fails
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params['api_key'] = api_key
    params['file_type'] = 'json'

    url = f"{BASE_URL}/{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"FRED API request failed: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"FRED error: {str(e)}")
        return None


def _get_series_latest(series_id: str) -> Optional[float]:
    """
    Get the most recent value for a FRED series.

    Args:
        series_id: FRED series identifier

    Returns:
        Latest value as float, or None if not available
    """
    params = {
        'series_id': series_id,
        'sort_order': 'desc',
        'limit': '1'
    }

    data = _make_request('series/observations', params)

    if data and 'observations' in data and len(data['observations']) > 0:
        value_str = data['observations'][0].get('value')
        if value_str and value_str != '.':
            try:
                return float(value_str)
            except ValueError:
                return None

    return None


def _get_series_yoy_change(series_id: str) -> Optional[float]:
    """
    Get year-over-year change for a FRED series.

    Args:
        series_id: FRED series identifier

    Returns:
        YoY change as percentage, or None if not available
    """
    # Get last 13 months of data to calculate YoY
    one_year_ago = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    params = {
        'series_id': series_id,
        'sort_order': 'desc',
        'observation_start': one_year_ago
    }

    data = _make_request('series/observations', params)

    if data and 'observations' in data and len(data['observations']) >= 2:
        obs = data['observations']

        # Get most recent and year-ago values
        latest_val = None
        year_ago_val = None

        for observation in obs:
            val_str = observation.get('value')
            if val_str and val_str != '.':
                try:
                    val = float(val_str)
                    if latest_val is None:
                        latest_val = val
                    else:
                        year_ago_val = val
                        break
                except ValueError:
                    continue

        if latest_val is not None and year_ago_val is not None and year_ago_val != 0:
            yoy_change = ((latest_val - year_ago_val) / year_ago_val) * 100
            return yoy_change

    return None


def get_risk_free_rate() -> float:
    """
    Get the current risk-free rate (10-Year Treasury yield).

    The 10-Year Treasury is the standard proxy for the risk-free rate
    used in CAPM and DCF valuations.

    Returns:
        10-Year Treasury yield as decimal (e.g., 0.043 for 4.3%)

        If API key not configured or data unavailable:
        Returns 0.043 (4.3%, reasonable default as of 2024-2025)
    """
    cache = get_cache()
    cache_key = "fred:risk_free_rate"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info("FRED risk-free rate cache hit")
        return cached

    # Check API key
    if not _get_api_key():
        logger.info("FRED not configured - using default risk-free rate of 4.3%")
        return 0.043

    logger.info("Fetching 10-Year Treasury yield from FRED")

    # DGS10 = 10-Year Treasury Constant Maturity Rate
    rate = _get_series_latest('DGS10')

    if rate is not None:
        # FRED returns percentage, convert to decimal
        rate_decimal = rate / 100

        # Cache for 24 hours
        cache.set(cache_key, rate_decimal, ttl_hours=24, source="fred")
        logger.info(f"Current risk-free rate: {rate:.2f}% ({rate_decimal:.4f})")

        return rate_decimal
    else:
        # Fallback to default
        logger.warning("Could not fetch risk-free rate, using default 4.3%")
        return 0.043


def get_macro_context() -> Dict[str, Any]:
    """
    Get macroeconomic context indicators.

    Provides key economic indicators that affect investment decisions:
    - GDP growth
    - Inflation (CPI)
    - Unemployment rate
    - Federal Funds rate (monetary policy)

    Returns:
        Dictionary containing:
        - gdp: {'value': float, 'yoy_change': float}
        - inflation: {'value': float, 'yoy_change': float}
        - unemployment: {'value': float, 'yoy_change': float}
        - fed_funds: {'value': float, 'yoy_change': float}

        If API key not configured, returns reasonable defaults:
        - GDP growth: 2.5%
        - Inflation: 3.2%
        - Unemployment: 4.0%
        - Fed Funds: 5.25%
    """
    cache = get_cache()
    cache_key = "fred:macro_context"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info("FRED macro context cache hit")
        return cached

    # Check API key
    if not _get_api_key():
        logger.info("FRED not configured - using default macro values")
        return {
            'gdp': {'value': 2.5, 'yoy_change': 0.3, 'note': 'Default value (API key not configured)'},
            'inflation': {'value': 3.2, 'yoy_change': -1.2, 'note': 'Default value (API key not configured)'},
            'unemployment': {'value': 4.0, 'yoy_change': -0.1, 'note': 'Default value (API key not configured)'},
            'fed_funds': {'value': 5.25, 'yoy_change': 0.5, 'note': 'Default value (API key not configured)'}
        }

    logger.info("Fetching macroeconomic context from FRED")

    macro = {}

    # GDP - Gross Domestic Product (billions, quarterly)
    # We'll use GDP growth rate instead
    gdp_val = _get_series_latest('GDP')
    gdp_yoy = _get_series_yoy_change('GDP')
    macro['gdp'] = {
        'value': gdp_val,
        'yoy_change': gdp_yoy,
        'unit': 'Billions of Dollars',
        'note': 'Quarterly data'
    }

    # CPI - Consumer Price Index (Inflation)
    cpi_val = _get_series_latest('CPIAUCSL')
    cpi_yoy = _get_series_yoy_change('CPIAUCSL')
    macro['inflation'] = {
        'value': cpi_val,
        'yoy_change': cpi_yoy,
        'unit': 'Index 1982-1984=100',
        'note': 'YoY change represents inflation rate'
    }

    # UNRATE - Unemployment Rate
    unrate_val = _get_series_latest('UNRATE')
    unrate_yoy = _get_series_yoy_change('UNRATE')
    macro['unemployment'] = {
        'value': unrate_val,
        'yoy_change': unrate_yoy,
        'unit': 'Percent',
        'note': 'Seasonally adjusted'
    }

    # FEDFUNDS - Federal Funds Effective Rate
    fedfunds_val = _get_series_latest('FEDFUNDS')
    fedfunds_yoy = _get_series_yoy_change('FEDFUNDS')
    macro['fed_funds'] = {
        'value': fedfunds_val,
        'yoy_change': fedfunds_yoy,
        'unit': 'Percent',
        'note': 'Monetary policy rate'
    }

    # Cache for 24 hours (macro data updates daily/weekly/monthly)
    cache.set(cache_key, macro, ttl_hours=24, source="fred")
    logger.info("Successfully fetched macro context from FRED")

    return macro


if __name__ == "__main__":
    # Quick test
    print("Testing FRED adapter...")

    rfr = get_risk_free_rate()
    print(f"\nRisk-Free Rate (10Y Treasury): {rfr*100:.2f}%")

    macro = get_macro_context()
    print(f"\nMacro Context:")
    print(f"  GDP: ${macro['gdp']['value']:.1f}B (YoY: {macro['gdp']['yoy_change']:.2f}%)")
    print(f"  Inflation: {macro['inflation']['yoy_change']:.2f}%")
    print(f"  Unemployment: {macro['unemployment']['value']:.1f}%")
    print(f"  Fed Funds: {macro['fed_funds']['value']:.2f}%")
