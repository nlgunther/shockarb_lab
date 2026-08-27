# Extending ShockArb's Factor Model to Medium/Long-Term Value

## 0. Correction: this is not a market-inefficiency bet

An earlier version of this memo treated Morningstar's Price/Fair-Value discount as the primary signal — the medium-term analogue of ShockArb's conf.Delta. That's wrong: it implicitly assumes the market has mispriced a stock relative to an "obvious" fair value, which is rare and not a sound basis for a long-horizon thesis. Markets are generally too efficient for large, persistent, obvious mispricings to sit around waiting to be found by reading a research report. 

The actual goal, per Ken (2026-08-18): investment is finding a promising management strategy at an attractive price — where "strategy" can be as simple as "keep doing the successful thing and keep getting better at it so no competitor can close the gap," not necessarily a dramatic pivot — and the whole exercise is worthless unless the thesis can be validated *now, and continuously, over time*. Otherwise the only way to learn whether you were right is watching the stock either soar or the company go bankrupt, which is far too late to be useful. You want to know as early as possible whether you were right or wrong, well before either terminal outcome.

That reframes what a large Price/Fair-Value gap is for: not a buy signal, but a **situation
finder** — evidence the market has already formed a strong view about some risk or opportunity, worth investigating to find out why. The work starts after that, not before.

## 1. The two pillars

Ken's distillation (2026-08-18), which supersedes the five-step version below it in this memo's history: an investment needs

1. **A significant margin of safety, usually from a moat or other barrier to entry.** The price paid needs room for the thesis to be wrong without catastrophic loss — this is the classic Graham/Buffett margin-of-safety idea, and a durable moat is what makes a margin of safety durable rather than a one-time accident of price.

2. **A management strategy transparent enough to inspect, evaluate, and — critically — to keep monitoring in close to real time and over time**, so the thesis can be validated or invalidated well before its terminal outcome (the stock re-rating, or the company failing) makes the answer obvious to everyone.

Pillar 1 is a valuation/safety question. Pillar 2 is an information-design question: can you get early, falsifiable signal on whether the strategy is working? Everything else in this memo — Five Forces, track record, segment residuals, disclosure checks — is a *tool* for evaluating one or both pillars, not a third pillar of its own.

### Two strategy archetypes, same discipline

The strategy being evaluated under pillar 2 comes in (at least) two shapes, and the leading indicators look different for each, though the discipline — name them before committing capital, check them every cycle, grade the outcome, revise belief — is identical:

- **Steady-state execution** ("keep doing the successful thing, better"). No inflection point to point to; the thesis is that a durable moat keeps widening or holding. Leading indicators here are the moat's own vital signs: market share trend, unit economics, customer retention/repeat-purchase rate, competitive response (are entrants actually succeeding or failing to dent the position), pricing power. A crack in these shows up long before a bad quarter does.

- **Strategic pivot / risk response** (Campbell's buying Rao's to offset declining soup). There's a specific before/after and a specific bet — leading indicators are tied to the pivot's own logic: is the new segment growing at the rate the price assumed, is the legacy segment's decline stabilizing or accelerating, is the capability required to execute the pivot showing up in adjacent evidence (management's track record on similar past moves).

## 2. Designing the leading indicators — the actual point of pillar 2

At the moment a thesis is formed, write down the specific, checkable things that would move you toward "confirmed" or "rejected" well before the stock price or an ultimate outcome tells you. Concretely, for each thesis:

1. **Name 3-5 falsifiable checkpoints**, each tied to a specific reporting event (next quarter's earnings, an investor day, a 10-K footnote) — not vague directional hopes. "Rao's grows" is not checkable; "Rao's consumption growth stays above X% for two more quarters" is.
2. **Decide in advance what confirms and what disconfirms**, including partial/ambiguous outcomes — a checkpoint that can only ever confirm the thesis isn't actually testing anything.
3. **Re-grade every cycle** (quarterly, post-earnings, for most of these theses) against the checkpoints set at initiation, not against a freshly re-derived story — the discipline only works if you're checking the same things you said you'd check, not moving the goalposts.
4. **Update the conviction/position size**, not just the narrative, when checkpoints disconfirm.

This is structurally the same thing `final_review` does for ShockArb every morning — verify claims against real, current evidence rather than trusting the story — just run on an earnings-cycle cadence instead of daily, and against pre-committed checkpoints instead of an ad hoc news check.

## 3. Supporting tools, mapped to the two pillars

**Five Forces + generic-strategy classification** (pillar 1, judging the moat). Threat of new entrants, bargaining power of suppliers/buyers, threat of substitutes, rivalry intensity, and whether the company's position is a coherent cost-leadership, differentiation, or focus strategy (vs. Porter's "stuck in the middle," usually the weakest position). This is how you judge whether the margin of safety is durable or just today's price.

**Disclosure-transparency check** (pillar 2, a precondition for setting any checkpoint at all). Flag any major strategic move (an acquisition >10% of enterprise value or >15x EBITDA at signing, a new segment, a stated turnaround) that loses standalone disclosure within a year or two of being announced — you can't set or grade a checkpoint on a number the company stops publishing.

**Segment-vs-peer residual** (pillar 2, finding and sizing the specific risk/opportunity to set checkpoints against). Regress a segment's trailing revenue growth and margin against a peer-group or category benchmark; a segment lagging its own category after controlling for the category's trend is a real, structural signal worth naming — this is what identified Campbell's soup segment as declining faster than its (mildly declining) category, and Rao's as outperforming its own.

**Track record on comparable past decisions** (pillar 1 and 2 both — informs how much moat/margin of safety a strategy deserves, and sets a prior for how the current checkpoints are likely to resolve). Has this management team executed a similar move successfully before?

**ROIC-vs-cost-of-capital test** (a specific checkpoint design for acquisition-funded pivots). Whether the incremental EBITDA/FCF an acquisition generates, divided by the price paid, clears the acquirer's cost of capital — independent of financing mix, since a dollar of cash spent has the same opportunity cost as a dollar of debt's interest rate.

**Source-agreement check** (a sanity check on pillar 1's price claim, not a substitute for either pillar). When a discount estimate comes from one shop's DCF, check it against the broader analyst community before treating "cheap" as fact rather than one model's opinion.

## 4. What this changes about the tracker

`wide_moat_tracker.xlsx`'s Core Candidates tabs (Low/Medium uncertainty, sorted by discount) are a reasonable situation-finder starting list — Low/Medium uncertainty roughly proxies for "transparent enough to actually set checkpoints on." The sort order (cheapest first) isn't a priority ranking, since cheapness isn't what's being optimized for; pick names by which risk/opportunity story is clearest and most checkable.

## 5. Applying the framework: Campbell's, with forward checkpoints

Run against real Q2/Q3 FY2026 numbers (see chat history for full sourcing).

**Pillar 1 (margin of safety / moat) — mixed.** Legacy soup: commoditized, high buyer power from retailer/private-label competition, thin differentiation — a hard position under Five Forces, not one the business is investing to defend. Rao's: a plausible differentiation-based moat (brand authenticity, less private-label substitution risk), but unproven at the scale and price paid ($2.7B, 14.6x/19.8x EBITDA). Net: whatever margin of safety exists rests almost entirely on Rao's, not on the legacy business, and Rao's moat is a thesis, not yet a fact.

**Pillar 2 (transparency) — fails on the specific thing that matters most.** Campbell folded Sovos/Rao's into an undisclosed "Distinctive Brands" unit inside Meals & Beverages after close (March 2024). That's not just an amber flag in the abstract — it specifically prevents setting a clean, checkable Rao's-standalone checkpoint, which is the single most important thing to be able to monitor here.

**Track record — the actual weak point.** The closest precedent, the older snacks portfolio (Pepperidge Farm, Goldfish, Kettle, from the 2018 Snyder's-Lance deal), is struggling right now: margin fell 390bp to 7% this quarter, management says recovery is "taking longer than anticipated." That's current, negative evidence on the exact capability the Rao's bet depends on.

**Forward checkpoints (set now, grade next quarter, FY26 Q4, expected ~October 2026):**

1. *Rao's growth rate.* Does consumption growth hold near the low-double-digits seen this quarter, or decelerate further toward the deal's implied requirement being missed? A drop toward high-single-digits for two consecutive quarters would materially weaken the pillar-1 case.
2. *Snacks margin trajectory.* Does margin show sequential recovery toward management's statedtarget, or continued deterioration? This directly tests the track-record concern — if snacksrecovers, it's evidence this management *can* fix an underperforming acquired branded-foodbusiness, which would also raise confidence in the Rao's bet by extension.
3. *Disclosure.* Does any standalone Sovos/Rao's figure reappear (a 10-K segment footnote, an investor-day breakout)? Reappearance would be a genuine positive transparency signal, not just neutral; continued opacity keeps pillar 2 failing.
4. *Soup category volume.* Does the -8% trend stabilize, worsen, or reflect a broader category decline (checkable against IRI/Nielsen-style category data, not just Campbell's own number)?
5. *Insider activity / buybacks.* Any change in insider buying or buyback pace at current levels would be a signal of management's own conviction (or lack of it) that's independent of the guidance they're giving publicly.

**Current verdict:** thesis leans negative — pillar 1's margin of safety depends on an unproven Rao's moat, pillar 2 fails on the one disclosure that matters most, and the closest read on management's track record for this specific capability is currently negative. Not a rejection; a "wait for checkpoints 1-2 to resolve before increasing conviction either way."

## 6. Concrete next step

1. Pick the next name and run the same process: name the strategy (steady-state or pivot), assess pillar 1 (Five Forces + track record) and pillar 2 (disclosure check), then **write the 3-5 forward checkpoints before doing anything else** — that's the step this memo was missing until today. Hershey (new this week: 5 stars, Low uncertainty, wide moat) is a reasonable next
   pick, same CPG space as Campbell's, so the comparison will be direct.
2. Log each name in a running file (`medium_term/VALUE_REVIEW.md`) structured around checkpoints, not a one-off verdict: initiation entry (pillars 1-2, checkpoints set), then a dated sub-entry each re-check grading each checkpoint confirmed/disconfirmed/ambiguous — the same append-only, dated-update convention `HIL_todo.md` already uses for ShockArb's recurring issues.
3. Re-check on the cadence each checkpoint's own reporting event dictates (usually quarterly, sometimes an investor day or 8-K) — not on a fixed calendar, since the point is to catch the checkpoint's actual resolution as early as possible, not to wait for a schedule.
