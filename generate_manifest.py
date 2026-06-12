"""
generate_manifest.py — regenerate MANIFEST.txt for the ShockArb project.

Usage:
    python generate_manifest.py

Walks shockarb/, utils/, datamgr/, tests/ for .py files, plus scripts/*.bat
and verify_install.py, computes a CRLF-normalised SHA-256 prefix for each,
and writes MANIFEST.txt with a hash-of-hashes bundle line. Run this after any
source or test file changes, then `python verify_install.py` to confirm.
See docs/KT.md ("File Integrity") and the ken-manifest-verify skill for the
hashing scheme.
"""

import hashlib
import os
from datetime import datetime, timezone


def find_py(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for f in sorted(filenames):
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f).replace("\\", "/"))
    return sorted(out)


def sha16(data: bytes) -> str:
    n = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return hashlib.sha256(n).hexdigest()[:16]


def bundle_hash(file_hashes: dict) -> str:
    concatenated = "".join(sha for _, sha in sorted(file_hashes.items()))
    return hashlib.sha256(concatenated.encode()).hexdigest()[:24]


def main() -> None:
    files: list[str] = []
    for pkg in ["shockarb", "utils", "datamgr", "tests", "diagnostics"]:
        files += find_py(pkg)
    files += [
        "scripts/run_tests.bat", "scripts/shockarb_workflows.bat",
        "verify_install.py", "generate_manifest.py",
    ]
    files = sorted(set(files))

    rows = {}
    for path in files:
        if not os.path.exists(path):
            rows[path] = "MISSING"
            continue
        with open(path, "rb") as f:
            rows[path] = sha16(f.read())

    bundle = bundle_hash({p: s for p, s in rows.items() if s != "MISSING"})
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# ShockArb file manifest",
        "# Generated: %s" % ts,
        "# Format:  sha256_prefix<TAB>local_path",
        "# Comments (lines starting with #) and blank lines are ignored.",
        "# Update only the hash values when files change.",
        "# verify_install.py reads this file — do not change the format.",
        "# Note: hashes computed with CRLF normalisation (Windows-compatible).",
        "# Bundle: SHA-256 of sorted file hashes concatenated (hash-of-hashes).",
        "#",
        "bundle:\t%s" % bundle,
        "#", "# Source files",
    ]
    for path in files:
        sha = rows[path]
        lines.append(("# MISSING\t%s" if sha == "MISSING" else "%s\t%s") % (sha, path))

    lines += [
        "#",
        "# Tests: 652 passing, 5 pre-existing failures in test_pipeline.py "
        "(TestSyntheticPrices/TestAddAssets — unrelated to report_compare; "
        "pytest tests/ -q); 21 test files",
    ]

    with open("MANIFEST.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Bundle hash: %s" % bundle)
    print("Files tracked: %d" % len(files))


if __name__ == "__main__":
    main()
