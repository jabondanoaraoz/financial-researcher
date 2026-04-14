"""
Workbook Generator
Entry point for Excel report generation.

Usage:
    from excel.workbook import generate_report
    generate_report(result, "output/AAPL_analysis.xlsx")

Author: Joaquin Abondano w/ Claude Code
"""

import os
import logging
from datetime import date

from openpyxl import Workbook

from .sheets.summary   import build as build_summary
from .sheets.analysts  import build as build_analysts
from .sheets.valuation import build as build_valuation
from .sheets.risk      import build as build_risk

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
        # Resolve output/ relative to this file's location (scripts/excel/)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out_dir  = os.path.join(base_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        today    = date.today().strftime("%Y%m%d")
        output_path = os.path.join(out_dir, f"{ticker}_{today}_analysis.xlsx")

    logger.info(f"[{ticker}] Generating Excel report → {output_path}")

    wb = Workbook()
    # Remove the default empty sheet
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    try:
        build_summary(wb, result)
        logger.info(f"[{ticker}] Sheet 1 (Summary) done.")
    except Exception as e:
        logger.error(f"[{ticker}] Summary sheet failed: {e}")

    try:
        build_analysts(wb, result)
        logger.info(f"[{ticker}] Sheet 2 (Analyst Panel) done.")
    except Exception as e:
        logger.error(f"[{ticker}] Analyst Panel sheet failed: {e}")

    try:
        build_valuation(wb, result)
        logger.info(f"[{ticker}] Sheet 3 (Valuation) done.")
    except Exception as e:
        logger.error(f"[{ticker}] Valuation sheet failed: {e}")

    try:
        build_risk(wb, result)
        logger.info(f"[{ticker}] Sheet 4 (Risk Profile) done.")
    except Exception as e:
        logger.error(f"[{ticker}] Risk Profile sheet failed: {e}")

    wb.save(output_path)
    logger.info(f"[{ticker}] Report saved → {output_path}")
    return output_path
