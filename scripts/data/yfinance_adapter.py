"""
YFinance Data Adapter
=====================
Main data provider for financial data extraction using yfinance library.

Author: Financial Researcher Team
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_company_info(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Get company information and metadata.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')

    Returns:
        Dictionary containing:
        - name: Company name
        - sector: Business sector
        - industry: Specific industry
        - market_cap: Market capitalization
        - currency: Trading currency
        - exchange: Stock exchange
        - description: Company description
        - website: Company website
        - employees: Number of employees
        - country: Country of origin

        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract and normalize company info
        company_info = {
            'name': info.get('longName') or info.get('shortName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': float(info.get('marketCap', 0)) if info.get('marketCap') else None,
            'currency': info.get('currency'),
            'exchange': info.get('exchange'),
            'description': info.get('longBusinessSummary'),
            'website': info.get('website'),
            'employees': int(info.get('fullTimeEmployees', 0)) if info.get('fullTimeEmployees') else None,
            'country': info.get('country')
        }

        logger.info(f"Successfully fetched company info for {ticker}")
        return company_info

    except Exception as e:
        logger.error(f"Failed to fetch company info for {ticker}: {str(e)}")
        return None


def get_prices(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Get historical price data.

    Args:
        ticker: Stock ticker symbol
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

    Returns:
        DataFrame with columns: Date (index), Open, High, Low, Close, Volume, Adj Close
        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        prices = stock.history(period=period, interval=interval)

        if prices.empty:
            logger.warning(f"No price data available for {ticker}")
            return None

        # Normalize column names
        prices.index.name = 'Date'

        # Ensure all numeric columns are float
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in prices.columns:
                prices[col] = prices[col].astype(float)

        logger.info(f"Successfully fetched {len(prices)} price records for {ticker}")
        return prices

    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {str(e)}")
        return None


def get_financials(ticker: str, years: int = 5) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Get financial statements.

    Args:
        ticker: Stock ticker symbol
        years: Number of years of historical data (default: 5)

    Returns:
        Dictionary containing three DataFrames:
        - income_statement: Annual + TTM income statement
        - balance_sheet: Annual + latest quarter balance sheet
        - cash_flow: Annual + TTM cash flow statement

        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)

        # Get annual financials
        income_stmt = stock.financials  # Annual
        balance = stock.balance_sheet  # Annual
        cash_flow = stock.cashflow  # Annual

        # Get quarterly data for TTM
        income_stmt_q = stock.quarterly_financials
        balance_q = stock.quarterly_balance_sheet
        cash_flow_q = stock.quarterly_cashflow

        # Limit to specified number of years
        if not income_stmt.empty and income_stmt.shape[1] > years:
            income_stmt = income_stmt.iloc[:, :years]
        if not balance.empty and balance.shape[1] > years:
            balance = balance.iloc[:, :years]
        if not cash_flow.empty and cash_flow.shape[1] > years:
            cash_flow = cash_flow.iloc[:, :years]

        # Combine annual with latest quarter/TTM
        financials = {
            'income_statement': income_stmt,
            'income_statement_quarterly': income_stmt_q,
            'balance_sheet': balance,
            'balance_sheet_quarterly': balance_q,
            'cash_flow': cash_flow,
            'cash_flow_quarterly': cash_flow_q
        }

        logger.info(f"Successfully fetched financial statements for {ticker}")
        return financials

    except Exception as e:
        logger.error(f"Failed to fetch financials for {ticker}: {str(e)}")
        return None


def get_key_metrics(ticker: str) -> Optional[Dict[str, float]]:
    """
    Get key valuation and performance metrics.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary containing:
        - pe_ratio: Price-to-Earnings ratio
        - pb_ratio: Price-to-Book ratio
        - ps_ratio: Price-to-Sales ratio
        - peg_ratio: PEG ratio
        - ev_ebitda: EV/EBITDA multiple
        - dividend_yield: Dividend yield (%)
        - beta: Stock beta
        - 52w_high: 52-week high
        - 52w_low: 52-week low
        - avg_volume: Average daily volume
        - shares_outstanding: Total shares outstanding
        - enterprise_value: Enterprise value
        - forward_pe: Forward P/E ratio
        - price_to_fcf: Price to Free Cash Flow

        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        metrics = {
            'pe_ratio': float(info.get('trailingPE', 0)) if info.get('trailingPE') else None,
            'pb_ratio': float(info.get('priceToBook', 0)) if info.get('priceToBook') else None,
            'ps_ratio': float(info.get('priceToSalesTrailing12Months', 0)) if info.get('priceToSalesTrailing12Months') else None,
            'peg_ratio': float(info.get('pegRatio', 0)) if info.get('pegRatio') else None,
            'ev_ebitda': float(info.get('enterpriseToEbitda', 0)) if info.get('enterpriseToEbitda') else None,
            'dividend_yield': float(info.get('dividendYield', 0)) * 100 if info.get('dividendYield') else None,
            'beta': float(info.get('beta', 0)) if info.get('beta') else None,
            '52w_high': float(info.get('fiftyTwoWeekHigh', 0)) if info.get('fiftyTwoWeekHigh') else None,
            '52w_low': float(info.get('fiftyTwoWeekLow', 0)) if info.get('fiftyTwoWeekLow') else None,
            'avg_volume': float(info.get('averageVolume', 0)) if info.get('averageVolume') else None,
            'shares_outstanding': float(info.get('sharesOutstanding', 0)) if info.get('sharesOutstanding') else None,
            'enterprise_value': float(info.get('enterpriseValue', 0)) if info.get('enterpriseValue') else None,
            'forward_pe': float(info.get('forwardPE', 0)) if info.get('forwardPE') else None,
            'price_to_fcf': float(info.get('priceToFreeCashFlow', 0)) if info.get('priceToFreeCashFlow') else None,
            'current_price': float(info.get('currentPrice', 0)) if info.get('currentPrice') else None,
        }

        logger.info(f"Successfully fetched key metrics for {ticker}")
        return metrics

    except Exception as e:
        logger.error(f"Failed to fetch key metrics for {ticker}: {str(e)}")
        return None


def get_news(ticker: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
    """
    Get recent news articles about the company.

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of news articles to return (default: 10)

    Returns:
        List of dictionaries containing:
        - title: Article title
        - publisher: News publisher
        - link: Article URL
        - publish_date: Publication timestamp

        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)

        # Try multiple methods to get news (yfinance API has changed over time)
        news_data = None

        # Method 1: Try .news attribute
        try:
            news_data = stock.news
        except:
            pass

        # Method 2: Try .get_news() method
        if not news_data:
            try:
                news_data = stock.get_news()
            except:
                pass

        if not news_data or len(news_data) == 0:
            logger.warning(f"No news available for {ticker} (this is normal - news API is often unavailable)")
            return []

        # Normalize news data
        news_list = []
        for article in news_data[:limit]:
            # Handle different possible key names
            title = article.get('title') or article.get('headline')
            publisher = article.get('publisher') or article.get('source')
            link = article.get('link') or article.get('url')
            pub_time = article.get('providerPublishTime') or article.get('publishedAt') or article.get('timestamp')

            news_item = {
                'title': title,
                'publisher': publisher,
                'link': link,
                'publish_date': datetime.fromtimestamp(pub_time) if pub_time else None
            }
            news_list.append(news_item)

        logger.info(f"Successfully fetched {len(news_list)} news articles for {ticker}")
        return news_list

    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {str(e)}")
        # Return empty list instead of None for news (news is optional)
        return []


def get_peer_suggestions(ticker: str) -> Optional[List[str]]:
    """
    Get suggested peer companies based on sector and industry.

    Args:
        ticker: Stock ticker symbol

    Returns:
        List of peer ticker symbols from the same sector/industry
        Returns None if data extraction fails
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get sector and industry
        sector = info.get('sector')
        industry = info.get('industry')

        if not sector:
            logger.warning(f"No sector information available for {ticker}")
            return None

        # Common peers by sector (this is a simplified approach)
        # In production, you would query a database or use the peers_mapping.json
        peers = []

        # Try to get recommendations from yfinance
        recommendations = info.get('recommendationKey')

        # For now, return an empty list with a note that peers should be loaded from peers_mapping.json
        logger.info(f"Sector: {sector}, Industry: {industry}")
        logger.info("Note: Peer suggestions should be loaded from templates/peers_mapping.json")

        return peers

    except Exception as e:
        logger.error(f"Failed to fetch peer suggestions for {ticker}: {str(e)}")
        return None


if __name__ == "__main__":
    # Quick test
    ticker = "AAPL"
    print(f"Testing yfinance adapter with {ticker}")

    info = get_company_info(ticker)
    if info:
        print(f"Company: {info['name']}")
