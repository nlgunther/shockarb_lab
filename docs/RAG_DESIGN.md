# ShockArb Local RAG — Design Document
*Status: proposed | Author: session 14 | Date: 2026-06-18*

---

## 1. What This Is

RAG (Retrieval-Augmented Generation) is a pattern in which a language model receives
a set of relevant past documents as part of its prompt at inference time, rather than
having that knowledge baked into its weights via training.  No fine-tuning occurs.
The model is the same every call; only the context changes.

For ShockArb the concrete operation is:

```
Today's market snapshot + live_alpha.csv
        ↓
   [Retriever] — searches reports/ index
        ↓
  3–5 past reports that are analytically similar to today
        ↓
  Gemini prompt:
    "Here are N past sessions where conditions were similar.
     Here is today's data.  Generate the narrative."
        ↓
   market_report.md / stock_report.md  (same as today)
```

The reports you have already generated (33 with `<!-- LEARN -->` tags) are the
initial corpus.  Every future `--timestamp` run extends it automatically.

---

## 2. Why RAG Rather Than Fine-Tuning

Fine-tuning a local Llama model on 33 examples would almost certainly overfit —
the model would memorize your specific phrasings rather than learn the analytical
pattern.  Fine-tuning becomes useful at roughly 500–2000 diverse examples.

RAG is superior here for three reasons:

1. **Immediate benefit.** The 33 reports are usable today without any training
   infrastructure or GPU.
2. **Transparent reasoning.** You can inspect exactly which past reports informed
   a narrative, and replace or remove them if they are stale.
3. **Updatable.** Each daily run adds a new example to the retrieval pool without
   any retraining cycle.

Fine-tuning remains a valid future path once the corpus exceeds ~500 reports and
a GPU is available.  The metadata schema defined in §4 is designed to support both.

---

## 3. Architecture

```
reports/
  ├── stock_report_20260612_0800.md          ← corpus document
  ├── stock_report_20260612_0800.meta.json   ← sidecar metadata (§4)
  ├── iran_shock/
  │   ├── stock_report_20260611_0122.md
  │   └── stock_report_20260611_0122.meta.json
  └── ...

utils/
  ├── rag/
  │   ├── __init__.py
  │   ├── indexer.py     ← writes .meta.json sidecars; builds reports/rag_index.json
  │   ├── retriever.py   ← filters + ranks; returns list[ReportMatch]
  │   └── context.py     ← renders matched reports into a prompt-ready string
  ├── marketfit/llm.py   ← modified: calls retriever before building prompt
  └── stockfit/llm.py    ← same
```

The index is a single flat JSON file (`reports/rag_index.json`) that aggregates
all `.meta.json` sidecars.  The retriever reads only the index (fast); the context
builder reads the full `.md` bodies of the top matches (slower, done at call time).

---

## 4. Report Metadata Schema

Each report receives a sidecar `.meta.json` file written by the CLI at generation
time.  The schema below is the complete specification.

```jsonc
{
  // ── Identity ──────────────────────────────────────────────────────────────
  "report_path":   "reports/stock_report_20260618_0813.md",
  "report_type":   "stock",          // "stock" | "market" | "stock_iran"
  "generated_at":  "2026-06-18T08:13:00Z",
  "regime":        "ukraine_shock",  // sticky regime at generation time
  "regime_phase":  "mid",            // "early" | "mid" | "late" | "normalization"
                                     // (see §4.1 for derivation rules)
  "has_llm":       true,             // false if --no-llm was used

  // ── Market structure (from market_snapshot.json) ──────────────────────────
  "market_verdict":  "GOOD",         // "GOOD" | "CAUTION" | "POOR"
  "market_score":    8,              // integer 0–11 from rules.evaluate()
  "vix":             14.2,
  "vix_regime":      "low",          // "low"(<15) | "normal"(15–25)
                                     //   | "elevated"(25–35) | "crisis"(>35)
  "spy_5d_pct":      1.8,            // SPY 5-day return %
  "spy_trend":       "up",           // "up" | "flat" | "down"
                                     //   (|5d%| < 0.5 → flat)
  "oil_price":       88.4,           // WTI spot; null if not in snapshot
  "oil_regime":      "elevated",     // "normal"(<80) | "elevated"(80–95)
                                     //   | "high"(>95); null if no oil data
  "tnx_yield":       4.31,           // 10Y treasury yield; null if absent
  "yield_trend":     "falling",      // "rising" | "flat" | "falling"

  // ── Signal quality (from live_alpha_*.csv) ────────────────────────────────
  "scores_file":     "live_alpha_us.csv",
  "include_count":   4,
  "watch_count":     3,
  "exclude_count":   59,
  "mean_r2_include": 0.71,           // mean r² across INCLUDE tickers
  "mean_conf_include": 0.038,        // mean confidence_delta across INCLUDE
  "top_tickers":     ["MSFT","AMAT","ADI","ETN"],
  "cluster_warnings": ["semiconductor_concentration"],
                                     // [] if none
  "coverage_drops":  0,              // count of tickers dropped for <80% coverage

  // ── Narrative phase (LLM-generated at report time, 1 sentence) ───────────
  "narrative_theme": "Tech mean-reversion in progress; energy stable; "
                     "semiconductor cluster dominates long side.",

  // ── Outcome (backfilled at T+5 trading days) ─────────────────────────────
  "outcome_date":    null,           // filled by scripts/backfill_outcomes.py
  "gap_recovery":    null,           // mean gap recovery across INCLUDE picks
  "outcome_label":   null            // 1 if gap_recovery >= 0.5, else 0
}
```

### 4.1 Deriving `regime_phase`

The regime phase is derived from the number of calendar days since the regime's
`universe.start_date`:

| Days since start | Phase           |
|-----------------|-----------------|
| 0 – 14          | `early`         |
| 15 – 45         | `mid`           |
| 46 – 90         | `late`          |
| > 90            | `normalization` |

These thresholds are heuristic and should be overridden manually when a clear
phase transition is observed (e.g. ceasefire announcement).  Override by editing
the sidecar directly; the indexer does not overwrite existing `regime_phase` values.

---

## 5. Retriever Logic

The retriever scores every indexed report against today's context and returns the
top-N by score.  Hard filters run first (disqualify unsuitable reports); soft
ranking runs on the survivors.

### 5.1 Hard filters

A report is excluded from retrieval if any of the following hold:

- `has_llm == false` — rules-based reports contain no useful narrative
- `report_type` does not match the current call type (`"stock"` vs `"market"`)
- Report is older than `MAX_AGE_DAYS` (default: 90)

### 5.2 Soft ranking (score = 0–100)

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Same regime | 40 | Regime determines factor structure; cross-regime analogies are weak |
| Same regime phase | 20 | Early-conflict dynamics differ sharply from normalization |
| Same VIX regime bin | 15 | Risk appetite shapes narrative tone |
| Same market verdict | 10 | GOOD vs POOR sessions require different narrative framing |
| Same SPY trend direction | 5 | Bull vs bear context |
| Same oil regime (if applicable) | 5 | Relevant for iran_shock; ignored for ukraine_shock |
| Recency (exponential decay, τ=30d) | 5 | Recent examples are more likely to be valid |

Top-3 reports by score are returned by default.  The caller may request up to 5.

### 5.3 Ticker overlap (stockfit only)

For `stockfit` retrieval, an additional tiebreaker: past reports that contain any
of today's INCLUDE tickers in their own `top_tickers` list score +5 per overlap.
This surfaces reports where the same names appeared previously, giving the LLM
richer per-ticker historical context.

---

## 6. Prompt Integration

The context builder renders matched reports into a block prepended to the existing
system prompt in `marketfit/llm.py` and `stockfit/llm.py`:

```
RETRIEVED CONTEXT — {N} similar past sessions
==============================================

[Session 1: 2026-06-12 | ukraine_shock/mid | GOOD | VIX 13.8]
{full body of stock_report_20260612_0800.md, truncated to MAX_CHARS_PER_DOC}

[Session 2: ...]
...
==============================================
Use the above sessions as examples of analytical style and signal interpretation.
Do not treat their conclusions as binding for today's data.
```

`MAX_CHARS_PER_DOC` defaults to 4 000 characters (~600 tokens), giving a 3-report
context block of ~1 800 tokens — well within Gemini's context window.

The existing narrative generation call in `llm.py` is unchanged.  The retriever
output is injected into the system message before the call.

---

## 7. Metadata That Does Not Yet Exist in Reports

The fields in §4 that require retrospective data or current enrichment are:

**`narrative_theme`** — currently absent. Add as a one-sentence LLM call at report
generation time (cheap: single sentence, no streaming).  The CLI should generate
this even when the main narrative is rules-based, so that `has_llm=false` reports
still contribute to the index with useful themes.

**`regime_phase`** — not in any current report or data file.  Must be derived at
index time from `regime.universe.start_date` (available via `shockarb.regimes`).

**`oil_price` / `oil_regime`** — partially present in `market_snapshot.json` as
an ETF price (USO or XLE), not as a WTI spot price.  The indexer can derive a
proxy from the USO close in the snapshot.

**`tnx_yield` / `yield_trend`** — present in `market_snapshot.json` as the TNX
field.  Needs extraction logic in the indexer.

**`outcome_*` fields** — entirely absent.  Require a separate
`scripts/backfill_outcomes.py` that runs weekly, looks up T+5 prices via yfinance,
and writes gap recovery to existing sidecars.  This is the path to eventually
training a logistic regression (the `labels.py` stub).

---

## 8. Implementation Plan

### Phase 1 — Index existing corpus (no code changes to LLM path)

1. Write `utils/rag/indexer.py`: scans `reports/**/*.md`, extracts metadata from
   filename, content footer stats, and `market_snapshot.json` snapshot (looked up
   by date), writes `.meta.json` sidecars and `rag_index.json`.
2. Run indexer over the 33 existing reports.  Manually fill `narrative_theme` for
   the first batch; auto-generate for all future reports.
3. Write `utils/rag/retriever.py` and `utils/rag/context.py`.
4. Add a `scripts/rag_index.py` CLI: `python scripts/rag_index.py --rebuild`.

### Phase 2 — Wire retriever into LLM calls

5. Modify `marketfit/llm.py`: call retriever before `generate_narratives()`;
   prepend context block to system message.
6. Modify `stockfit/llm.py`: same, with ticker-overlap tiebreaker.
7. Add `--no-rag` flag to both CLIs for debugging (default: RAG on when index
   exists).

### Phase 3 — Auto-metadata at generation time

8. Modify `marketfit/cli.py` and `stockfit/cli.py`: after saving `.md`, generate
   and save `.meta.json` sidecar automatically.
9. Add the one-sentence `narrative_theme` LLM call to the sidecar generation.

### Phase 4 — Outcome backfill (enables future ML)

10. Write `scripts/backfill_outcomes.py`: weekly cron, fills `outcome_*` fields.
11. Once `outcome_label` is populated for ≥30 reports, implement `labels.py`.

---

## 9. Metadata Reflection — What Future Reports Should Capture

The fields below represent what would make retrieval most powerful over time.
They are ranked by implementation cost vs. retrieval value.

### High value, low cost (add in Phase 3)

- `regime`, `regime_phase`, `market_verdict`, `market_score` — derivable from
  existing pipeline outputs at zero extra API cost.
- `vix`, `vix_regime`, `spy_5d_pct`, `spy_trend` — already in
  `market_snapshot.json`; just need extraction.
- `include_count`, `watch_count`, `top_tickers` — already in report footer.
- `coverage_drops` — already in `shockarb.log`.

### High value, moderate cost (add in Phase 3, requires one LLM call)

- `narrative_theme` — a single-sentence characterization of the analytical
  challenge of the day.  This is the most powerful retrieval signal because it
  captures the *type of reasoning* required, not just the surface conditions.
  Example: *"Geopolitical premium fading; watch for energy mean-reversion as
  strait risk recedes."*  Cost: one short LLM call per report.

- `dominant_sector` — which sector drove the INCLUDE list (derived from ticker
  cluster annotations already computed by `rules.py`).  Surfacing semiconductor
  vs. energy vs. defense concentration helps retrieve structurally similar sessions.

### Moderate value, moderate cost (consider for Phase 3/4)

- `oil_regime`, `tnx_yield`, `yield_trend` — present in snapshot but need
  extraction.  High value for iran_shock retrieval; moderate value for
  ukraine_shock.
- `geopolitical_phase` — a human-maintained enum per regime instance:
  `escalation | peak_stress | plateau | normalization | resolution`.  Not
  derivable from data; requires one line of manual annotation per session.
  Very high retrieval power; low automation potential.

### Lower value or high cost (defer)

- Full embedding vectors for semantic similarity — would require a local
  embedding model or API call per report.  Overkill given the structured
  metadata available; revisit if corpus exceeds 200 reports.
- News headline embeddings — interesting but adds complexity; the `news.txt`
  content is already available to the LLM in context.
- Intraday volatility metrics — useful for timing analysis but not for the
  narrative-generation use case.

### The single most important addition

If only one metadata field is added to every future report, it should be
`narrative_theme`.  Structured fields like VIX and regime match reports with
similar *conditions*; `narrative_theme` matches reports with similar *analytical
challenges*.  Two sessions can have identical VIX and regime but very different
interpretive problems — one a clean trending signal, another a noisy
cross-regime conflict.  The theme captures that distinction in a form both
humans and the retriever can use.
