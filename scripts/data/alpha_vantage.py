"""
Alpha Vantage Data Adapter
===========================
Enhanced financial metrics and technical indicators from Alpha Vantage API.

Alpha Vantage provides pre-calculated financial ratios, analyst ratings,
and technical indicators that complement basic yfinance data.

Free Tier: 25 requests/day
API Docs: https://www.alphavantage.co/documentation/

Author: Financial Researcher Team
"""

import requests
import os
from typing import Optional, Dict, Any
import logging
import pandas as pd
from datetime import datetime

from data.cache import get_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.alphavantage.co/query"


def _get_api_key() -> Optional[str]:
    """Get Alpha Vantage API key from environment."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key or api_key == "your_alpha_vantage_key_here":
        logger.warning("Alpha Vantage API key not configured - features disabled")
        return None

    return api_key


def _make_request(params: Dict[str, str], timeout: int = 30) -> Optional[Dict]:
    """
    Make a request to Alpha Vantage API.

    Args:
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        JSON response as dict, or None if request fails
    """
    api_key = _get_api_key()
    if not api_key:
        return None

    params['apikey'] = api_key

    try:
        response = requests.get(BASE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        # Check for API error messages
        if 'Error Message' in data:
            logger.error(f"Alpha Vantage API error: {data['Error Message']}")
            return None

        if 'Note' in data:
            # Rate limit message
            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
            return None

        return data

    except requests.exceptions.RequestException as e:
        logger.error(f"Alpha Vantage request failed: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"Alpha Vantage error: {str(e)}")
        return None


def get_company_overview(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get comprehensive company overview with pre-calculated ratios.

    Provides fundamental data and valuation metrics that are pre-calculated
    by Alpha Vantage, including analyst ratings and target prices.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing:
        - Valuation: PERatio, PEGRatio, BookValue, PriceToBookRatio, etc.
        - Dividends: DividendPerShare, DividendYield, DividendDate, ExDividendDate
        - Profitability: EPS, ProfitMargin, OperatingMarginTTM, ROE, ROA
        - Growth: QuarterlyEarningsGrowthYOY, QuarterlyRevenueGrowthYOY
        - Analyst: AnalystTargetPrice, AnalystRating breakdown

        Returns None if:
        - API key not configured (graceful degradation)
        - Ticker not found
        - API request fails
    """
    cache = get_cache()
    cache_key = f"alpha_vantage:overview:{ticker.upper()}"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Alpha Vantage overview cache hit for {ticker}")
        return cached

    # Check API key
    if not _get_api_key():
        logger.info("Alpha Vantage not configured - skipping company overview")
        return None

    logger.info(f"Fetching Alpha Vantage company overview for {ticker}")

    params = {
        'function': 'OVERVIEW',
        'symbol': ticker.upper()
    }

    data = _make_request(params)

    if not data or not data.get('Symbol'):
        logger.warning(f"No overview data available for {ticker}")
        return None

    try:
        # Extract and normalize key metrics
        overview = {
            # Identification
            'symbol': data.get('Symbol'),
            'name': data.get('Name'),
            'exchange': data.get('Exchange'),
            'sector': data.get('Sector'),
            'industry': data.get('Industry'),

            # Valuation Ratios
            'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio') and data.get('PERatio') != 'None' else None,
            'peg_ratio': float(data.get('PEGRatio', 0)) if data.get('PEGRatio') and data.get('PEGRatio') != 'None' else None,
            'price_to_book': float(data.get('PriceToBookRatio', 0)) if data.get('PriceToBookRatio') and data.get('PriceToBookRatio') != 'None' else None,
            'price_to_sales': float(data.get('PriceToSalesRatioTTM', 0)) if data.get('PriceToSalesRatioTTM') and data.get('PriceToSalesRatioTTM') != 'None' else None,
            'ev_to_revenue': float(data.get('EVToRevenue', 0)) if data.get('EVToRevenue') and data.get('EVToRevenue') != 'None' else None,
            'ev_to_ebitda': float(data.get('EVToEBITDA', 0)) if data.get('EVToEBITDA') and data.get('EVToEBITDA') != 'None' else None,

            # Per Share Metrics
            'book_value': float(data.get('BookValue', 0)) if data.get('BookValue') and data.get('BookValue') != 'None' else None,
            'eps': float(data.get('EPS', 0)) if data.get('EPS') and data.get('EPS') != 'None' else None,
            'revenue_per_share_ttm': float(data.get('RevenuePerShareTTM', 0)) if data.get('RevenuePerShareTTM') and data.get('RevenuePerShareTTM') != 'None' else None,

            # Dividends
            'dividend_per_share': float(data.get('DividendPerShare', 0)) if data.get('DividendPerShare') and data.get('DividendPerShare') != 'None' else None,
            'dividend_yield': float(data.get('DividendYield', 0)) if data.get('DividendYield') and data.get('DividendYield') != 'None' else None,
            'dividend_date': data.get('DividendDate'),
            'ex_dividend_date': data.get('ExDividendDate'),

            # Profitability
            'profit_margin': float(data.get('ProfitMargin', 0)) if data.get('ProfitMargin') and data.get('ProfitMargin') != 'None' else None,
            'operating_margin_ttm': float(data.get('OperatingMarginTTM', 0)) if data.get('OperatingMarginTTM') and data.get('OperatingMarginTTM') != 'None' else None,
            'return_on_assets_ttm': float(data.get('ReturnOnAssetsTTM', 0)) if data.get('ReturnOnAssetsTTM') and data.get('ReturnOnAssetsTTM') != 'None' else None,
            'return_on_equity_ttm': float(data.get('ReturnOnEquityTTM', 0)) if data.get('ReturnOnEquityTTM') and data.get('ReturnOnEquityTTM') != 'None' else None,

            # Size
            'revenue_ttm': float(data.get('RevenueTTM', 0)) if data.get('RevenueTTM') and data.get('RevenueTTM') != 'None' else None,
            'gross_profit_ttm': float(data.get('GrossProfitTTM', 0)) if data.get('GrossProfitTTM') and data.get('GrossProfitTTM') != 'None' else None,
            'market_cap': float(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization') and data.get('MarketCapitalization') != 'None' else None,

            # Growth
            'quarterly_earnings_growth_yoy': float(data.get('QuarterlyEarningsGrowthYOY', 0)) if data.get('QuarterlyEarningsGrowthYOY') and data.get('QuarterlyEarningsGrowthYOY') != 'None' else None,
            'quarterly_revenue_growth_yoy': float(data.get('QuarterlyRevenueGrowthYOY', 0)) if data.get('QuarterlyRevenueGrowthYOY') and data.get('QuarterlyRevenueGrowthYOY') != 'None' else None,

            # Analyst Ratings
            'analyst_target_price': float(data.get('AnalystTargetPrice', 0)) if data.get('AnalystTargetPrice') and data.get('AnalystTargetPrice') != 'None' else None,
            'analyst_strong_buy': int(data.get('AnalystRatingStrongBuy', 0)) if data.get('AnalystRatingStrongBuy') and data.get('AnalystRatingStrongBuy') != 'None' else 0,
            'analyst_buy': int(data.get('AnalystRatingBuy', 0)) if data.get('AnalystRatingBuy') and data.get('AnalystRatingBuy') != 'None' else 0,
            'analyst_hold': int(data.get('AnalystRatingHold', 0)) if data.get('AnalystRatingHold') and data.get('AnalystRatingHold') != 'None' else 0,
            'analyst_sell': int(data.get('AnalystRatingSell', 0)) if data.get('AnalystRatingSell') and data.get('AnalystRatingSell') != 'None' else 0,
            'analyst_strong_sell': int(data.get('AnalystRatingStrongSell', 0)) if data.get('AnalystRatingStrongSell') and data.get('AnalystRatingStrongSell') != 'None' else 0,

            # Other
            'beta': float(data.get('Beta', 0)) if data.get('Beta') and data.get('Beta') != 'None' else None,
            '52_week_high': float(data.get('52WeekHigh', 0)) if data.get('52WeekHigh') and data.get('52WeekHigh') != 'None' else None,
            '52_week_low': float(data.get('52WeekLow', 0)) if data.get('52WeekLow') and data.get('52WeekLow') != 'None' else None,
        }

        # Cache for 24 hours (fundamentals don't change frequently)
        cache.set(cache_key, overview, ttl_hours=24, source="alpha_vantage")
        logger.info(f"Successfully fetched Alpha Vantage overview for {ticker}")

        return overview

    except Exception as e:
        logger.error(f"Failed to parse Alpha Vantage overview for {ticker}: {str(e)}")
        return None


def get_technical_indicators(ticker: str, days: int = 60) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Get technical indicators for stock analysis.

    Provides key technical indicators used in momentum and trend analysis.

    Args:
        ticker: Stock ticker symbol
        days: Number of recent days to return (default: 60)

    Returns:
        Dictionary containing DataFrames for:
        - rsi: Relative Strength Index (14-period)
        - macd: Moving Average Convergence Divergence
        - bbands: Bollinger Bands (20-period)
        - sma_50: 50-day Simple Moving Average
        - sma_200: 200-day Simple Moving Average

        Returns None if:
        - API key not configured (graceful degradation)
        - Ticker not found
        - API request fails
    """
    cache = get_cache()
    cache_key = f"alpha_vantage:technicals:{ticker.upper()}:{days}d"

    # Check cache
    cached = cache.get(cache_key)
    if cached:
        logger.info(f"Alpha Vantage technicals cache hit for {ticker}")
        return cached

    # Check API key
    if not _get_api_key():
        logger.info("Alpha Vantage not configured - skipping technical indicators")
        return None

    logger.info(f"Fetching Alpha Vantage technical indicators for {ticker}")

    indicators = {}

    try:
        # RSI (Relative Strength Index)
        rsi_params = {
            'function': 'RSI',
            'symbol': ticker.upper(),
            'interval': 'daily',
            'time_period': '14',
            'series_type': 'close'
        }
        rsi_data = _make_request(rsi_params)

        if rsi_data and 'Technical Analysis: RSI' in rsi_data:
            rsi_series = rsi_data['Technical Analysis: RSI']
            rsi_df = pd.DataFrame.from_dict(rsi_series, orient='index')
            rsi_df.index = pd.to_datetime(rsi_df.index)
            rsi_df = rsi_df.sort_index(ascending=False).head(days)
            rsi_df.columns = ['RSI']
            rsi_df['RSI'] = rsi_df['RSI'].astype(float)
            indicators['rsi'] = rsi_df

        # Note: Alpha Vantage has strict rate limits (25 req/day on free tier)
        # To avoid exhausting the limit, we would need to make ONE request
        # and get multiple indicators, or use a premium tier.
        # For now, we'll only fetch RSI to conserve requests.

        logger.info("Note: Only fetching RSI to conserve Alpha Vantage API quota (25 req/day)")
        logger.info("MACD, Bollinger Bands, and SMAs would require additional API calls")

    except Exception as e:
        logger.error(f"Failed to fetch technical indicators for {ticker}: {str(e)}")
        return None

    if indicators:
        # Cache for 1 hour (technicals update daily but we want fresh data)
        cache.set(cache_key, indicators, ttl_hours=1, source="alpha_vantage")
        logger.info(f"Successfully fetched technical indicators for {ticker}")
        return indicators
    else:
        return None


if __name__ == "__main__":
    # Quick test
    print("Testing Alpha Vantage adapter...")

    ticker = "AAPL"

    overview = get_company_overview(ticker)
    if overview:
        print(f"\nCompany: {overview['name']}")
        print(f"P/E Ratio: {overview['pe_ratio']}")
        print(f"Analyst Target: ${overview['analyst_target_price']}")

    technicals = get_technical_indicators(ticker)
    if technicals and 'rsi' in technicals:
        print(f"\nLatest RSI: {technicals['rsi'].iloc[0]['RSI']:.2f}")
