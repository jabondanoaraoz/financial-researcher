"""
Workbook Generator  (v3)
Entry point for Excel report generation.

Usage:
    from excel.workbook import generate_report
    generate_report(result, "output/AAPL_analysis.xlsx")

Sheet order:
    1. Financials     — company profile, income statement, cash flow, balance sheet
    2. Multiples      — peer-based valuation with user-editable weights
    3. DCF Model      — full IB-style DCF with Excel formulas
    4. Analyst Panel  — all 10 agents with scoring, pillar breakdown, reasoning
    5. Summary        — investment decision, consensus, risk, PM thesis

Author: Joaquin Abondano w/ Claude Code
"""

import os
import logging
from datetime import date

from openpyxl import Workbook

from .sheets.financials  import build as build_financials
from .sheets.multiples   import build as build_multiples
from .sheets.dcf         import build as build_dcf
from .sheets.analysts    import build as build_analysts
from .sheets.summary     import build as build_summary

logger = logging.getLogger(__name__)


def generate_report(result: dict, output_path: str = None) -> str:
    """
    Generate a multi-sheet Excel report from a run_analysis() result dict.

    Parameters
    ----------
    result : dict
        Output of orchestrator.run_analysis().
    output_path : str | None
        Destination .xlsx path. If None, auto-generates under output/.

    Returns
    -------
    str
        Absolute path to the saved file.
    """
    ticker = result.get("ticker", "TICKER")

    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir  = os.path.join(base_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        today    = date.today().strftime("%Y%m%d")
        output_path = os.path.join(out_dir, f"{ticker}_{today}_analysis.xlsx")

    logger.info(f"[{ticker}] Generating Excel report → {output_path}")

    wb = Workbook()
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    sheets = [
        ("Financials",    build_financials),
        ("Multiples",     build_multiples),
        ("DCF Model",     build_dcf),
        ("Analyst Panel", build_analysts),
        ("Summary",       build_summary),
    ]

    for name, builder in sheets:
        try:
            builder(wb, result)
            logger.info(f"[{ticker}] Sheet '{name}' done.")
        except Exception as e:
            logger.error(f"[{ticker}] Sheet '{name}' failed: {e}", exc_info=True)

    wb.save(output_path)
    logger.info(f"[{ticker}] Report saved → {output_path}")
    return output_path
