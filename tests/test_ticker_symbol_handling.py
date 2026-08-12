import unittest

import pytest

from cli.utils import normalize_ticker_symbol
from tradingagents.agents.utils.agent_utils import (
    astock_special_reports_context,
    build_instrument_context,
    detect_instrument_market,
    filter_analysts_for_market,
    market_scope_context,
)


@pytest.mark.unit
class TickerSymbolHandlingTests(unittest.TestCase):
    def test_normalize_ticker_symbol_preserves_exchange_suffix(self):
        self.assertEqual(normalize_ticker_symbol(" cnc.to "), "CNC.TO")

    def test_build_instrument_context_mentions_exact_symbol(self):
        context = build_instrument_context("7203.T")
        self.assertIn("7203.T", context)
        self.assertIn("exchange suffix", context)

    def test_detect_instrument_market_from_ticker_shape(self):
        self.assertEqual(detect_instrument_market("600519"), "a_stock")
        self.assertEqual(detect_instrument_market("NVDA"), "us_stock")

    def test_us_stock_filters_a_share_only_analysts(self):
        analysts = ["market", "news", "policy", "hot_money", "lockup", "fundamentals"]
        self.assertEqual(
            filter_analysts_for_market(analysts, "us_stock"),
            ["market", "news", "fundamentals"],
        )

    def test_us_market_scope_does_not_penalize_missing_astock_reports(self):
        context = market_scope_context("AAPL")
        self.assertIn("US stock", context)
        self.assertIn("not as hidden negative evidence", context)

        reports = astock_special_reports_context("AAPL")
        self.assertIn("intentionally skipped", reports)
        self.assertIn("Do not infer bearish risk", reports)

    def test_astock_special_reports_are_included_for_a_share(self):
        reports = astock_special_reports_context(
            "600519",
            policy_report="policy data",
            hot_money_report="flow data",
            lockup_report="lockup data",
        )
        self.assertIn("Policy Analysis Report", reports)
        self.assertIn("Hot Money / Capital Flow Report", reports)
        self.assertIn("Lockup Expiry / Insider Reduction Report", reports)


if __name__ == "__main__":
    unittest.main()
