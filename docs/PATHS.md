# ShockArb Path Conventions

This document explains how file paths are managed across the ShockArb pipeline:
why paths are centralised, why `pathlib.Path` objects are used instead of
strings, why paths are relative rather than absolute, and what the working-
directory convention means in practice.

---

## Design Decisions

### 1. Single source of truth: `utils/paths.py`

All pipeline file paths — both inputs and outputs — are defined once in
`utils/paths.py`. No other module carries its own path literals.

**Why this matters:** when a path changes (e.g. the reports folder is
renamed, or inputs are moved to a subdirectory), you change one line in
`paths.py`. Nothing else needs to touch. Scattered path strings across
multiple files are a maintenance hazard and a source of subtle bugs when one
copy is updated and another is not.

### 2. `pathlib.Path` objects, not strings

`paths.py` exports `Path` objects. Callers import and use them directly —
no `str()` casting is ever needed.

**Why no `str()` casts?** Python's standard library — `open()`,
`read_text()`, `write_text()`, `os.path.*`, `argparse` defaults — accepts
`Path` objects natively. Casting a `Path` to `str` before passing it to
these functions is unnecessary overhead and strips the semantic type
information. If you see `str(SOME_PATH)` in the codebase, that is a bug to
fix, not a pattern to copy.

### 3. Relative paths, not absolute

Paths are defined relative to the working directory at runtime — not
relative to `paths.py`'s own location on disk.

**Why relative?** Absolute paths computed from `__file__` (e.g.
`Path(__file__).resolve().parent.parent / "data"`) are robust in theory but
fragile in practice: they break when the project is moved, symlinked, or
accessed from a network share. Relative paths keep the project fully
portable — moving the project folder requires no code changes, only the
working-directory convention (see below).

**The trade-off:** relative paths require a consistent working directory.
This is enforced by convention and by a startup check in the CLI entry
points.

---

## Working-Directory Convention

**All `stockfit` and `marketfit` commands must be run from the `utils/`
directory.**

```bash
cd <project_root>/utils
python -m stockfit report
python -m marketfit report
```

From `utils/`, the relative paths resolve correctly:

| Path constant       | Resolves to (from `utils/`)         |
|---------------------|-------------------------------------|
| `DATA`              | `<project_root>/data/`              |
| `REPORTS`           | `<project_root>/reports/`           |
| `LIVE_ALPHA_US`     | `<project_root>/data/live_alpha_us.csv` |
| `FUNDAMENTALS`      | `<project_root>/data/fundamentals.csv`  |
| `NEWS`              | `<project_root>/data/news.txt`          |
| `MARKET_SNAPSHOT`   | `<project_root>/data/market_snapshot.json` |
| `MARKET_REPORT`     | `<project_root>/reports/market_report.md`  |
| `MARKET_REPORT_INTRADAY` | `<project_root>/reports/market_report_intraday.md` |
| `STOCK_REPORT`      | `<project_root>/reports/stock_report.md`   |
| `REPORTS_DIR`       | `<project_root>/reports/`           |

The `scripts/shockarb_workflows.bat` file enforces the project root via
`cd /d "%~dp0.."` before calling any pipeline command, satisfying this
convention automatically.

---

## CWD Error Detection

Both `stockfit` and `marketfit` CLIs check the working directory on startup
and emit a clear error if the convention is not met:

```
❌  This command must be run from the utils/ directory.

    Correct usage:
        cd <project_root>/utils
        python -m stockfit report

    Current directory: C:\Users\me\some\other\place
```

The check verifies that `../data/` exists as a directory relative to CWD.
This is a project-specific marker that will not be present in an arbitrary
working directory.

---

## Adding or Moving Paths

To add a new named path:

```python
# In utils/paths.py
MY_NEW_FILE = DATA / "my_new_file.csv"
```

Then import it wherever needed:

```python
from paths import MY_NEW_FILE
```

To relocate the inputs folder (e.g. to `data/pipeline_inputs/`), change
the `DATA` or add a separate `_INPUTS` alias in `paths.py`. Nothing else
needs to touch.

---

## What Not To Do

```python
# ❌ Do not define paths outside paths.py
_DEFAULT_SCORES = "../data/live_alpha_us.csv"

# ❌ Do not cast Path objects to str
default=str(LIVE_ALPHA_US)

# ❌ Do not use __file__-relative absolute paths
ROOT = Path(__file__).resolve().parent.parent
```

All three patterns have appeared in this codebase during development and
have been deliberately removed. If you see them again, fix them here.
