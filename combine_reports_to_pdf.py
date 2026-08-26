"""
combine_reports_to_pdf.py - Combine one or more ShockArb Markdown reports
into a single PDF "briefing book".

Concatenates the given Markdown files, in the order given, each starting on
a new page, and converts the result to PDF via pandoc + xelatex.

Why the symbol handling: ShockArb reports are full of emoji and the price-
direction triangles (up/down arrows) used in every table row. xelatex's
default Latin Modern fonts don't include most of them, producing dozens of
"Missing character" warnings and visible empty boxes in the PDF (root-caused
2026-08-06). Decorative section-header emoji are dropped outright. The
up/down triangles are dropped too - every occurrence in these reports is
immediately followed by a signed percentage (e.g. "+0.44%"), so the arrow
never carries information the sign doesn't already give. The >= comparison
operator is a different case: it appears in report threshold lines (e.g.
"r-squared >= 0.65") and DOES carry meaning, so it's replaced with an ASCII
equivalent rather than dropped.

Requires: pandoc and xelatex on PATH. If missing: pandoc.org for pandoc; any
TeX distribution that includes xelatex (TeX Live, MiKTeX) for the engine.

Usage:
    python combine_reports_to_pdf.py report1.md report2.md -o out.pdf

    python combine_reports_to_pdf.py ^
        reports\market_report_2026-08-07_0851.md ^
        reports\stock_report_20260807_1147.md ^
        reports\final_review_20260807_0925.md ^
        -o briefing_book_US_equities_20260807.pdf
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Decorative glyphs dropped outright - no information lost (see module
# docstring for the up/down-triangle reasoning specifically).
_DROP = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"  # regional indicator (flag) letters
    "\U0001F300-\U0001FAFF"  # misc emoji blocks
    "☀-➿"          # misc symbols & dingbats (warning sign, check/cross mark, etc.)
    "▲▼"           # up/down price-direction triangles
    "️"                 # variation selector-16
    "]"
)

# Glyphs that carry real meaning - replaced with an ASCII equivalent
# instead of dropped. Add more here the same way if a future report uses
# another symbol xelatex's default fonts don't cover.
_REPLACE = {
    "≥": ">=",  # >=
}


def strip_unsupported_symbols(text: str) -> str:
    """
    Clean text of glyphs xelatex's default (Latin Modern) fonts can't
    render, so pandoc -> xelatex produces a PDF with no missing-character
    boxes.

    Example:
        strip_unsupported_symbols("## [chart emoji] Report\\n[up-arrow] +0.44%\\nr-sq >= 0.65")
        # -> "## Report\\n +0.44%\\nr-sq >= 0.65"
    """
    for char, replacement in _REPLACE.items():
        text = text.replace(char, replacement)
    text = _DROP.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^ +", "", text, flags=re.MULTILINE)
    return text


def combine(paths: list[Path]) -> str:
    """
    Read and concatenate `paths`, cleaned of unsupported symbols, with a
    page break inserted before each file after the first.

    Example:
        combine([Path("a.md"), Path("b.md")])
        # -> "<a's cleaned content>\\n\\n```{=latex}\\n\\\\newpage\\n```\\n\\n<b's cleaned content>"
    """
    parts = []
    for i, path in enumerate(paths):
        text = strip_unsupported_symbols(path.read_text(encoding="utf-8"))
        if i > 0:
            parts.append("```{=latex}\n\\newpage\n```\n")
        parts.append(text)
    return "\n\n".join(parts)


def convert_to_pdf(markdown_text: str, out_path: Path) -> None:
    """
    Run `markdown_text` through pandoc + xelatex, writing `out_path`.
    Raises RuntimeError with pandoc's stderr on failure.
    """
    with tempfile.TemporaryDirectory() as tmp:
        combined_md = Path(tmp) / "combined.md"
        combined_md.write_text(markdown_text, encoding="utf-8")
        result = subprocess.run(
            [
                "pandoc", str(combined_md), "-o", str(out_path),
                "--pdf-engine=xelatex",
                "-V", "geometry:margin=1in",
                "-V", "fontsize=10pt",
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed:\n{result.stderr}")
        missing = [l for l in result.stderr.splitlines() if "Missing character" in l]
        if missing:
            print(
                f"Warning: {len(missing)} missing-character warning(s) from xelatex - "
                "some glyph in the input isn't covered by strip_unsupported_symbols() yet. "
                "First one:\n  " + missing[0],
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine Markdown reports into a single PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("reports", nargs="+", type=Path,
                         help="Markdown report files, in the order they should appear")
    parser.add_argument("-o", "--out", type=Path, required=True, help="Output PDF path")
    args = parser.parse_args()

    missing = [p for p in args.reports if not p.exists()]
    if missing:
        print(f"Error: file(s) not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        sys.exit(1)

    combined = combine(args.reports)
    convert_to_pdf(combined, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
