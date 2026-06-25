# LLM Todo — shockarb_lab

> Items for Claude to action in a future session.
> Format: `- [ ] REFERENCE — what to do — any relevant context`

---

## Open Items

### Enhancements: market_data.py / market report

- [ ] **MARKET-HOLIDAY-FLAG** — Add US market holiday awareness to `market_data.py` so the stale-data warning is suppressed when `last_date` is the last valid *trading* day. When a holiday caused the skip, the market report should note it explicitly: "Last US trading session: {date} ({holiday name} — market closed {skipped date})." Use `pandas_market_calendars` (NYSE calendar) or a static holiday list. Also update the market report template to render the holiday note in the ⚠️ header block if present in the snapshot.

### Enhancements: fundamental_scanner.py

- [ ] **fetch_fundamentals** — Add leverage metrics to the fundamentals table — e.g. debt-to-equity (`debtToEquity` from yf.Ticker.info), possibly also net debt / EBITDA if available.

---

## Completed Items

*(Move entries here once done.)*
