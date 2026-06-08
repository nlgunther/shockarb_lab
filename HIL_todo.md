# HIL Todo — shockarb_lab

> Items requiring human verification or judgment before proceeding.
> Format: `- [ ] TICKER — issue description — suggested fix`

---

## Open Items

### Data Quality: Industry Classifications

The industry field in score reports is sourced from an external reference (likely NASDAQ/NYSE listing data or yfinance). Several tickers are systematically misclassified.

- [ ] **PH** — Review source of industry assignment. Was "Office Equipment/Supplies/Services"; corrected to "Industrial Machinery/Components". Verify source and fix upstream if possible.
- [ ] **LHX** — L3Harris is a defense electronics/EW systems company. Currently shows "Industrial Machinery/Components" (same bucket as ETN, ROK). Should likely be "Aerospace" or a defense-specific category. Skipped in auto-fix to avoid collateral corrections to other valid Industrial Machinery entries.
- [ ] **CPRT** — Copart is an online vehicle auction platform, not a traditional auto dealer. "Services-Auto (Vehicle Auctions)" is a workaround; verify the canonical SIC/NASDAQ category.
- [ ] **QCOM** — Tagged "Semiconductors/Wireless Technology" as a compromise. Source data said "Computer peripheral equipment". Confirm correct category vs. peer group (ADI, TXN, KLAC are all "Semiconductors").
- [ ] **HII** — Huntington Ingalls is a shipbuilder; corrected from "Metal Fabrications" to "Aerospace". Confirm whether "Shipbuilding" is a distinct category in the source data.

### Performance Analysis: Executive Overview

- [ ] **BACKFILL-SCORES** — Need weekly score CSVs (Dec 2024 – May 2026) to reconstruct 6-month signal series. Run the backfill loop below on your machine and drop results into `data/backfill/`. — See command in conversation (2026-05-31).
- [ ] **BACKTEST-OUTPUT** — Need `shockarb backtest --model data\ukraine_shock_us_20260528_143030.json --trailing-window 130 --return-type both --holding-periods 1 2 3 5 --top-n 5 --min-r-squared 0.50 --min-confidence 0.005` output. Paste or save to `data/backfill/backtest_output.txt`. — Required for T+1…T+5 mean returns and hit rate table.

---

## Resolved Items

*(Move entries here once verified and fixed upstream.)*

---

## Notes

- Source of industry strings: likely `ticker_reference_cache.json` or the NASDAQ/NYSE CSV listings in `data/`. Check `shockarb/names.py` for the lookup chain.
- If the source is NASDAQ/NYSE CSVs, industry strings come from their "Sector" or "Industry" column and may lag or be coarse.
