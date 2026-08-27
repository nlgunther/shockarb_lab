# KT — Medium/Long-Term Value Investing Track

> Last updated: 2026-08-21T13:15-04:00 | Trigger: manual | Staleness: Drifting

**Drifting note (2026-08-21):** five sections rewritten this pass (§1, §3, §4, §6, §7) — not undetected drift, but a full work session completing since the last update: Hershey was run through the two-pillar process, `VALUE_REVIEW.md` was created, and a direct Campbell's-vs-Hershey comparison was written. Content below is current as of this update.

---

## 1. Project Overview

A second, separate investing track from ShockArb, worked out of `medium_term/`: medium/long-term value investing on multi-year holdings, using a two-pillar framework (moat/margin of safety + a monitorable strategy) rather than a mispricing bet. Started 2026-08-18, prompted by Ken's sense that ShockArb's short-term mean-reversion edge may be decaying. Current status: **Active Development** — framework is built and has now been applied end-to-end to two names (Campbell's, Hershey), logged in `VALUE_REVIEW.md` with a direct comparison between them; both remain in "wait for checkpoints" status, not action-ready.

## 2. Goals & Constraints

**Goals:**
- Build a small number of high-conviction, multi-year value positions using a repeatable, two-pillar process (margin of safety via moat; a strategy transparent enough to monitor in real time).
- Get early, falsifiable signal on each thesis via pre-committed forward checkpoints, so the thesis can be validated or rejected well before the stock re-rates or the company fails — not just at the terminal outcome.

**Constraints:**
- The framework itself is explicitly unfinished and meant to be refined iteratively as it's applied to more names — not to be treated as a frozen spec.
- Weekly Morningstar "Weekly Highlights" PDFs (dropped into `medium_term/` by Ken) are the situation-finder input; `wide_moat_tracker.xlsx` is transcribed from these by hand each week.
- Analysis only — no live trading or execution happens in this track.

**Non-goals:**
- Not a discount/mispricing-screening strategy — a large Price/Fair-Value gap is a situation finder, never a buy signal.
- Not a slower version of ShockArb — explicitly not a mean-reversion bet.

## 3. Prototypes & Examples

**Campbell's (CPB)** — full pillar 1/2 run completed 2026-08-18. Screen data: 5 stars, Medium uncertainty, Wide moat, FV $56 vs. price $23.17, $6.9B market cap (Morningstar, Aug 14 screen). Pillar 1 (moat) mixed: legacy soup has no real moat and is being harvested; Rao's has a plausible differentiation-based moat but unproven at the ~$2.7B / 14.6x–19.8x EBITDA price paid. Pillar 2 (transparency) fails: Rao's results are folded into an undisclosed "Distinctive Brands" unit within Meals & Beverages, blocking a clean Rao's-standalone checkpoint. Track record on the closest precedent (Snyder's-Lance-era snacks: Pepperidge Farm, Goldfish) is currently negative — margin down 390bp to 7% this quarter, recovery "taking longer than anticipated" per management. Five forward checkpoints set for FY26 Q4 (~Oct 2026): Rao's growth rate, snacks margin trajectory, disclosure reappearance, soup category volume, insider buying/buybacks. **Verdict: lean negative — wait for checkpoints to resolve, not a rejection.** As of 2026-08-21: unchanged, checkpoints not yet due. Full write-up: `VALUE_REVIEW.md`.

**Hershey (HSY)** — full pillar 1/2 run completed 2026-08-21. Screen data: 5 stars, Low uncertainty, Wide moat, FV $230 vs. price $184.22 (Morningstar, Aug 14 screen); live price $186.46 as of 2026-08-21. Pillar 1 (moat) real and currently demonstrated: Q2 FY2026 revenue $2.79B (+6.6% YoY, beat by 5.8%), EBIT beat by 26.4%, driven by ~10 points of net price realization — the moat visibly passing a real cocoa cost shock through to the customer. Live crack to watch: ~2-point volume decline alongside that pricing; management itself frames the open question as temporary elasticity vs. early moat erosion. Pillar 2 (transparency) **passes**, unlike Campbell's: the ~$3B Salty Snacks bet (Dot's Pretzels, SkinnyPop, LesserEvil) is reported as its own standalone segment with real sales and income each quarter ($388M net sales +23% YoY, but segment income -5.9% YoY on freight/production bottlenecks in Q2). Five forward checkpoints set for Q3 FY2026 (~late Oct/Nov 2026): Salty Snacks margin recovery, core volume trend, cocoa cost guidance, consensus-vs-screen convergence (Street consensus is Hold, ~$205-217, notably below Morningstar's $230), insider activity. **Verdict: lean constructive, more so than Campbell's specifically on pillar 2** — the diversification bet is fully checkpoint-able, where Campbell's is not. Full write-up and the direct Campbell's-vs-Hershey comparison: `VALUE_REVIEW.md`.

## 4. Architecture & Key Files

- `value_factor_model_extension.md` — the living framework memo: two-pillar model, checkpoint-design discipline, supporting tools (Five Forces, disclosure check, segment-vs-peer residual, ROIC-vs-cost-of-capital, source-agreement check) mapped to the pillar each serves. Updated in place as the methodology is refined — not versioned or archived. Per-name analysis has now moved out of this file and into `VALUE_REVIEW.md` (see below); this memo stays focused on the framework itself.
- `wide_moat_tracker.xlsx` — situation-finder screen, transcribed weekly from Morningstar's Weekly Highlights PDFs. Convention: each week adds a new dated sheet pair (`Full Screen (Aug N)`, `Core Candidates (Aug N)`) rather than overwriting the prior week — currently holds Aug 7 and Aug 14 pairs. Core Candidates is Low/Medium-uncertainty names sorted by discount; the sort order is a starting list, not a priority ranking (cheapness isn't what's being optimized for).
- `mstar0810_1426.pdf`, `mstar08126.pdf` — raw Morningstar source PDFs Ken drops in the folder; inputs to the xlsx transcription.
- `VALUE_REVIEW.md` — **created 2026-08-21.** The append-only, per-name running log: initiation entry with pillars 1–2 + checkpoints, then dated re-check sub-entries grading each checkpoint, same convention `HIL_todo.md` uses for ShockArb. Currently holds Campbell's (migrated from the framework memo) and Hershey (new), plus a dedicated "Campbell's vs. Hershey" comparison section.

## 5. Recent Decisions & Rationale

- **2026-08-18** — Rejected "find stocks trading irrationally below fair value" as the operating premise; adopted the two-pillar framework instead. *Why:* markets are too efficient for large, persistent, obvious mispricings; Ken wants early falsifiability on a strategy thesis, not a bet that only resolves at the terminal outcome.
- **2026-08-18** — Ran Campbell's through the framework; verdict lean negative, checkpoints set for FY26 Q4. *Why:* pillar 2 fails on the one disclosure that matters most (Rao's folded into an undisclosed unit), and the closest management track record on this specific capability is currently negative.
- **2026-08-18** — Selected Hershey as the next name, over other Core Candidates entries. *Why:* same CPG space as Campbell's enables a direct comparison, and it screens with lower uncertainty (Low vs. Campbell's Medium).
- **2026-08-18** — `wide_moat_tracker.xlsx` convention set to additive dated sheet pairs rather than overwrite-in-place. *Why:* Ken wants to see the screen evolve week over week, not just the latest snapshot.
- **2026-08-21** — Ran Hershey through the framework; verdict lean constructive, checkpoints set for Q3 FY2026. *Why:* pillar 2 passes cleanly (Salty Snacks reported as a real standalone segment), and the core confectionery business is demonstrating real pricing power through an active cocoa cost shock — a materially stronger starting position than Campbell's declining soup core.
- **2026-08-21** — Created `VALUE_REVIEW.md` and migrated Campbell's out of the framework memo; added Hershey and a dedicated Campbell's-vs-Hershey comparison section. *Why:* this was the planned running-log convention from 2026-08-18, executed once a second name existed to compare against.
- **2026-08-21** — Comparison finding: despite screening as statistically "cheaper" (larger Price/FV discount), Campbell's is judged the weaker setup of the two under this framework, because its diversification bet is undisclosed and unmonitorable while Hershey's is fully checkpoint-able. *Why:* this is the framework's central thesis in action — cheap-and-opaque is worse than moderately-priced-and-transparent, since only the latter allows early falsifiability. Worth remembering as a general lesson for future names, not just this pair.

## 6. Open Questions & Blockers

1. ~~Hershey has not been run through the two-pillar process yet~~ — resolved 2026-08-21, see `VALUE_REVIEW.md`.
2. Campbell's five FY26 Q4 checkpoints can't be graded until Campbell's reports (~Oct 2026). Added 2026-08-18.
3. ~~`VALUE_REVIEW.md` hasn't been created yet~~ — resolved 2026-08-21, created and populated with both names plus the comparison.
4. Hershey's five Q3 FY2026 checkpoints can't be graded until Hershey reports (~late Oct/early Nov 2026, based on the Q2 report's July 30 date). Added 2026-08-21.
5. No third candidate name has been picked yet from `wide_moat_tracker.xlsx`'s Core Candidates list. Added 2026-08-21.

## 7. Next Steps

1. Pick a third name from `wide_moat_tracker.xlsx`'s Core Candidates list; check the disclosure-transparency question (pillar 2) before going deep on Five Forces, per the lesson from the Campbell's-vs-Hershey comparison.
2. Re-check Campbell's five checkpoints against actual FY26 Q4 results when reported (~Oct 2026).
3. Re-check Hershey's five checkpoints against Q3 FY2026 results when reported (~late Oct/early Nov 2026).
4. Continue weekly Morningstar transcription into `wide_moat_tracker.xlsx` as new PDFs arrive.

## 8. Last Updated

2026-08-21T13:15-04:00 | Trigger: manual (Ken asked to find "disappeared" value-investing discussions, write them + a Campbell's-vs-Hershey comparison to `medium_term/`, and refresh this KT) | Staleness: Drifting (5 sections rewritten — expected forward progress, not undetected drift) | Summary: confirmed nothing had actually been lost from disk (the framework memo and this KT both survived untouched); ran Hershey through the full two-pillar process for the first time, created `VALUE_REVIEW.md` with both names plus a direct comparison, and updated this document to match.
