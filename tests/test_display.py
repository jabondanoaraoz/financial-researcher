"""
Terminal display layer test - no API calls required.
Run from project root: python tests/test_display.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from display import print_summary
from mock_result import make_mock_result


if __name__ == "__main__":
    result = make_mock_result()
    print_summary(result)
