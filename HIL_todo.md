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

### Data Quality: live_alpha_us.csv / RVOL cache (2026-06-11)

- [ ] **FTNT row truncated** — `data/live_alpha_us.csv` row 65 (FTNT, last row) has only 8 of 9 columns; the trailing `confidence_delta` value is missing, so `csv.DictReader` reads it as `None`. — `stockfit/features.py` (`_load_scores`/`_load_fundamentals`) now treats missing/`None` fields as empty/NaN so `python -m stockfit report` no longer crashes, but FTNT's `confidence_delta` is NaN and it's excluded from scoring as a result. Re-run `shockarb score` to regenerate `live_alpha_us.csv` cleanly and confirm FTNT gets a real confidence_delta.
- [x] **RVOL shows "—" for all tickers** — RESOLVED 2026-06-11. Root cause was NOT the sandbox-parquet issue noted below — Ken's local cache (`data/prices/daily/{ETN,HON,ISRG}.parquet`, 171 rows each, current through 2026-06-10) was healthy. The real bug: `_compute_rvol()` in `stockfit/features.py` looked for column `("Volume", ticker)`, but the actual cached parquet schema uses LOWERCASE column names (`volume`, `adj_close`, `adj_factor`, `close`, `high`, `low`, `open`) — confirmed via a one-off diagnostic script. `_compute_rvol` now checks for both `"Volume"` and `"volume"`. Added regression test `test_lowercase_volume_column_supported` (tests/test_stockfit_rvol.py). 593 tests pass. Ken should re-run `python -m stockfit report --rvol` to confirm real RVOL values now populate. (The sandbox-mount "Parquet magic bytes not found" symptom seen earlier this session was a separate, sandbox-only artifact and is not related to this fix.)

### LLM Narrative Quality: stock_report (2026-06-11 20:43)

- [ ] **CPRT** — `--llm` narrative in `reports/stock_report_20260611_2043.md` calls CPRT "Coinbase" ("Coinbase's price of $31.06 reflects an overreaction..."). CPRT is Copart (vehicle salvage auctions), not Coinbase (COIN). The LLM appears to be confusing the ticker with an unrelated company name. — Check `utils/stockfit/llm.py` prompt: is company name passed in, or is the LLM inferring it from the ticker alone? If this report feeds the training corpus (`docs/corpus/`), this bad narrative should be excluded/corrected before use.
- [ ] **MSFT/V cluster tagging** — Both MSFT and V are annotated "Mega-Cap Tech" cluster risk in the same report. Visa is a payments/financial-services name, not tech — grouping it with MSFT under "Mega-Cap Tech" looks like a sector-classification bug in the cluster annotation logic (`stockfit/rules.py`). — Verify sector source and cluster grouping logic; may share root cause with the industry-classification issues above.
- [x] **Large analyst-upside figures (MSFT +43.7%, CPRT +33.4%, V +25.0%)** — CHECKED 2026-06-11. `data/fundamentals.csv` is internally consistent (Fwd P/E recomputes correctly from Price/Fwd EPS for all three: MSFT 20.2x, CPRT 18.4x, V 21.5x). `data/analyst_overrides.csv` has no entries for MSFT/CPRT/V (only KLAC/QCOM/ISRG), so these targets are raw yfinance consensus, not a parsing bug. The implied re-rating is large (target P/E ~25-29x vs current ~18-22x for all three), which is consistent with analyst targets lagging a sharp price selloff rather than bad data — but that lag means the "upside" may shrink once analysts revise targets down, not because price reverts up. — Treat these upside figures as potentially stale/lagging, not as a hard data error. Re-check after next `fundamental_scanner` run / analyst revision cycle.

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
