"""
Excel report generation test — no API calls required.
Run from project root: python tests/test_excel.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from excel.workbook import generate_report
from mock_result import make_mock_result


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    result = make_mock_result()
    path = generate_report(result, "output/AAPL_test.xlsx")
    print(f"Excel report saved → {path}")
