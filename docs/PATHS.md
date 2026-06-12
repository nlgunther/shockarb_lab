# ShockArb Path Conventions

This document explains how file paths are managed across the ShockArb pipeline:
why paths are centralised, why `pathlib.Path` objects are used instead of
strings, and how the project-root anchor works in practice.

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

### 3. `__file__`-relative root, not CWD-relative

The project root is computed once, at import time, from `paths.py`'s own
location on disk:

```python
_ROOT = Path(__file__).parent.parent   # utils/paths.py → utils/ → shockarb_lab/
DATA    = _ROOT / "data"
REPORTS = _ROOT / "reports"
```

`Path(__file__)` is the absolute path to `utils/paths.py` itself.
`.parent` gives `utils/`.
`.parent.parent` gives `shockarb_lab/` — the project root.

All path constants derived from `_ROOT` are therefore **absolute paths**
that never depend on the current working directory.

**Why not CWD-relative?** Earlier versions used `Path("../data")` which
required all commands to run from `utils/`. When `shockarb_workflows.bat`
runs commands like `python -m shockarb score` from the project root, the
relative `../data` resolved to the *parent* of the project root, causing
`live_alpha_us.csv` to be written to the wrong location. The `__file__`
anchor eliminates this class of bug entirely.

---

## Working-Directory Convention

**Scripts may be invoked from any directory.** `shockarb_workflows.bat`
always `cd`s to the project root at the top, and all Python commands run
from there:

```bat
REM Always run from the project root (one level above scripts\)
cd /d "%~dp0.."
python -m shockarb score
python utils\news_scanner.py
```

Because `paths.py` uses `__file__`-anchored constants, the working directory
no longer matters for path resolution.

---

## Path Constants

| Constant                  | Resolves to                                          |
|---------------------------|------------------------------------------------------|
| `_ROOT`                   | `<project_root>/`                                    |
| `DATA`                    | `<project_root>/data/`                               |
| `REPORTS`                 | `<project_root>/reports/`                            |
| `LIVE_ALPHA_US`           | `<project_root>/data/live_alpha_us.csv`              |
| `FUNDAMENTALS`            | `<project_root>/data/fundamentals.csv`               |
| `NEWS`                    | `<project_root>/data/news.txt`                       |
| `MARKET_SNAPSHOT`         | `<project_root>/data/market_snapshot.json`           |
| `MARKET_REPORT`           | `<project_root>/reports/market_report.md`            |
| `MARKET_REPORT_INTRADAY`  | `<project_root>/reports/market_report_intraday.md`   |
| `STOCK_REPORT`            | `<project_root>/reports/stock_report.md`             |
| `REPORTS_DIR`             | `<project_root>/reports/`                            |
| `STOCKFIT_RVOL_FILE`      | `<project_root>/data/.stockfit_rvol`                 |

---

## Sticky CLI State

`STOCKFIT_RVOL_FILE` (`data/.stockfit_rvol`) follows the same sticky-file
pattern as `shockarb`'s `.shockarb_regime`: a small text file in `data/`
holding a literal `on` or `off`, written by `stockfit set-rvol <on|off>` and
read by `stockfit report` to decide whether to compute/display the RVOL
column. Resolution order is `--rvol` / `--no-rvol` flag > sticky file >
default `off`. See `docs/KT.md` ("RVOL (relative volume) display") for the
full RVOL design.

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
`DATA` or add a separate `_INPUTS` alias in `paths.py`. Nothing else
needs to touch.

---

## What Not To Do

```python
# ❌ Do not define paths outside paths.py
_DEFAULT_SCORES = "../data/live_alpha_us.csv"

# ❌ Do not cast Path objects to str
default=str(LIVE_ALPHA_US)

# ❌ Do not use CWD-relative paths in paths.py
DATA = Path("../data")   # breaks when called from project root
```

The CWD-relative `Path("../data")` pattern appeared in this codebase and
was removed because it caused `shockarb score` to write `live_alpha_us.csv`
to the wrong directory when run from the project root. All paths must be
anchored to `_ROOT`.
