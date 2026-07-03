---
name: hil-followup
description: >
  Maintain a HIL_todo.md file in any project folder to track items that require human
  verification, judgment, or upstream correction before they can be fully resolved.
  Use this skill whenever Claude encounters data quality issues, ambiguous classifications,
  uncertain corrections, or any finding where the right answer requires human lookup or
  decision — and the work should continue rather than block. Triggers include: "add to
  HIL todo", "flag this for review", "note this for later", "create a HIL entry", or
  whenever Claude makes a best-effort correction that should be verified. One HIL_todo.md
  per project root. Do not use for blocking issues (use hil-practice for those).
---

# HIL Followup Log

This skill manages `HIL_todo.md` — a persistent, human-readable list of items that
need human attention but do not block the current task. The distinction from
`hil-practice` is important:

- **hil-practice** — Claude *stops* and waits. The issue blocks forward progress.
- **hil-followup** — Claude *continues*, logs the issue, and moves on. A human
  resolves it later at their own pace.

---

## When to Use This Skill

Invoke this skill whenever any of the following arise mid-task:

1. **Best-effort corrections** — Claude fixes something with reasonable confidence
   but cannot verify the canonical answer (e.g., correcting a misclassified industry
   from public knowledge, not from the source data).

2. **Data quality findings** — A value in a file, report, or dataset looks wrong but
   fixing it upstream requires access Claude doesn't have (a database, an API, a
   vendor feed).

3. **Skipped fixes** — Claude identifies an error but deliberately skips the
   auto-fix because it would cause collateral damage to other valid rows/entries.

4. **Ambiguous corrections** — Multiple plausible corrections exist; Claude picks one
   but the right answer requires domain judgment or external lookup.

5. **Source tracing needed** — The fix is clear but the root cause (which file,
   which pipeline step, which config) needs a human to trace and repair permanently.

---

## File Location and Naming

- One file per project: `HIL_todo.md` in the project root.
- Never create HIL_todo files in subdirectories or module folders.
- If HIL_todo.md does not exist, create it using the template in the next section.
- If it already exists, **append** new items — never rewrite existing open items.

---

## Document Structure

HIL_todo.md must follow this exact structure. Do not deviate without good reason.

```markdown
# HIL Todo — <Project Name>

> Items requiring human verification or judgment before proceeding.
> Format: `- [ ] REFERENCE — issue description — suggested fix or next step`

---

## Open Items

### <Category> (e.g., "Data Quality: Industry Classifications")

- [ ] **REFERENCE** — What is wrong and why it matters — Suggested fix or where to look

### <Another Category>

- [ ] ...

---

## Resolved Items

*(Move entries here once verified and fixed upstream.)*

---

## Notes

*(Context about data sources, lookup paths, or recurring patterns.)*
```

### Section rules

**`## Open Items`**
- Required. Always present, even if empty.
- Items are grouped under `###` category headers.
- Add a new `###` category when a new class of issue appears.
- Never delete an open item — either resolve it (move to Resolved) or mark it
  `[x]` with a brief resolution note on the same line.

**`## Resolved Items`**
- Required. Starts empty with the placeholder comment.
- When resolving an item, move it here and append a short note:
  `- [x] **PH** — Industry corrected upstream in ticker_reference_cache.json. (2026-05-30)`

**`## Notes`**
- Optional but encouraged. Use for:
  - Source file paths or lookup chains relevant to the category
  - Patterns that explain *why* multiple items share the same root cause
  - Links to relevant tickets, docs, or references

---

## Item Format

Each item is one line:

```
- [ ] **REFERENCE** — <what is wrong> — <suggested fix or where to look>
```

- **REFERENCE**: ticker symbol, filename, function name, config key, or any
  short identifier that makes the item scannable.
- **What is wrong**: one sentence. State the observed value and what it should be.
- **Suggested fix or where to look**: one sentence. Be specific — name the file,
  field, or API if known. If unknown, say so.

Keep items atomic — one issue per line. Do not combine multiple tickers into one
item unless they share the exact same root cause *and* the same fix location.

---

## Update Protocol

When adding items during a task:

1. Check whether HIL_todo.md exists in the project root. Create it if not.
2. Identify the right `###` category for the new item. Create the category if needed.
3. Append the item under that category. Do not re-sort or reformat existing items.
4. If the item is a skipped auto-fix, note what was skipped and why (collateral
   damage risk, ambiguity, etc.) so the human reviewer has full context.
5. After all items are added, summarize in the response: "Added N items to HIL_todo.md."

When a human says an item is resolved:

1. Move it from Open Items to Resolved Items.
2. Append a one-line resolution note and today's date in parentheses.
3. Do not delete the item — the Resolved section is a permanent audit trail.

---

## Cross-Project Use

This skill is project-agnostic. It works identically in shockarb_lab,
StatementGuard, CorpusForge, paytracker, or any other project folder Ken uses.
The only thing that changes is the project name in the `# HIL Todo — <Project Name>`
header. All structural rules apply in every project.
