"""
Financial Researcher - CLI entry point.

Usage:
    python run.py AAPL
    python run.py AAPL MSFT GOOGL
    python run.py AAPL --peers MSFT GOOGL AMZN
    python run.py AAPL --no-excel
    python run.py AAPL --output reports/AAPL_custom.xlsx
    python run.py AAPL --quiet
"""

import argparse
import logging
import os
import sys
from datetime import date

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from orchestrator import run_analysis
from display import print_summary
from excel.workbook import generate_report


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run a full 10-agent stock analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("tickers", nargs="+", type=str, help="One or more ticker symbols (e.g. AAPL MSFT GOOGL)")
    p.add_argument(
        "--peers", nargs="+", metavar="TICKER",
        help="Custom peer universe (e.g. --peers MSFT GOOGL AMZN)",
    )
    p.add_argument(
        "--no-excel", action="store_true",
        help="Skip Excel report generation",
    )
    p.add_argument(
        "--output", metavar="PATH",
        help="Custom output path for Excel file (default: output/<TICKER>_<date>.xlsx)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress terminal summary (still generates Excel unless --no-excel)",
    )
    p.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING)",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    tickers = [t.upper().strip() for t in args.tickers]
    peers   = {t.upper(): None for t in args.peers} if args.peers else None
    total   = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        if total > 1:
            print(f"\n[{i}/{total}] Analyzing {ticker}...")
        else:
            print(f"\nAnalyzing {ticker}...")

        result = run_analysis(ticker, peers=peers)

        if not args.quiet:
            print_summary(result)

        if not args.no_excel:
            if args.output and total == 1:
                output_path = args.output
            else:
                os.makedirs("output", exist_ok=True)
                output_path = os.path.join("output", f"{ticker}_{date.today()}.xlsx")

            generate_report(result, output_path)
            print(f"Excel report saved → {output_path}\n")

    if total > 1:
        print(f"Done. {total} reports generated in output/\n")


if __name__ == "__main__":
    main()
