"""
ShockArb End-of-Day Scanner.

Loads the saved factor model(s), fetches today's closing returns, scores the
tape, and exports the results to CSV.  Run this once after the 4pm close to
produce the alpha sheets consumed by the other utils.

Output files
------------
    data/live_alpha_us.csv      (if a US model exists)
    data/live_alpha_global.csv  (if a Global model exists)

Usage examples
--------------
    # Scan both universes (default)
    python utils/daily_scanner.py

    # Scan US only
    python utils/daily_scanner.py --universe us

    # Scan with a custom data directory
    python utils/daily_scanner.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
from email import parser
import os
import sys

import pandas as pd
import yfinance as yf
from loguru import logger

from shockarb.config import ExecutionConfig
import shockarb.pipeline as pipeline


# =============================================================================
# Live return fetcher
# =============================================================================




# =============================================================================
# Scanner
# =============================================================================

def run_scanner(universes, exec_cfg, force_daily=False, from_open=False) -> None:
    """Score and export one or more universes."""
    # Universe registry — maps CLI name to UniverseConfig.
    # Add new universes here as they are built and saved.
    from shockarb.config import US_UNIVERSE, GLOBAL_UNIVERSE
    UNIVERSES = {
        "us":     US_UNIVERSE,
        "global": GLOBAL_UNIVERSE,
    }

    any_ran = False

    for name in universes:
        print(f"\n{'='*80}")
        print(f"  SCANNING: {name.upper()} MODEL")
        print(f"{'='*80}")

        universe = UNIVERSES.get(name)
        if universe is None:
            logger.error(f"Unknown universe {name!r}. Known: {sorted(UNIVERSES)}")
            continue

        model_path = pipeline.find_latest_model(name, exec_cfg)
        if not model_path:
            logger.error(f"No saved model for '{name}'. Run: python -m shockarb build --universe {name}")
            continue

        model = pipeline.load_model(model_path)

        try:
            scores, prov = pipeline.score_universe(universe, model, exec_cfg,
                                        force_daily=force_daily,
                                        from_open=from_open)
            prov.model_file = model_path
        except ValueError as exc:
            logger.error(f"Failed to score '{name}': {exc}")
            continue

        # Print provenance to console
        print(f"\n{prov.summary()}\n")

        # Save scores CSV
        output_path = os.path.join(exec_cfg.data_dir, f"live_alpha_{name}.csv")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        scores.to_csv(output_path)
        logger.success(f"Alpha sheet saved: {output_path}")

        # Save provenance sidecar
        import json
        prov_path = os.path.join(exec_cfg.data_dir, f"live_alpha_{name}_provenance.json")
        with open(prov_path, "w") as f:
            json.dump(prov.to_dict(), f, indent=2)
        logger.success(f"Provenance saved: {prov_path}")

        any_ran = True

    if any_ran:
        print("\n✅ Scan complete.")
        print("   Next steps:")
        print("   • python utils/news_scanner.py              (headlines for top signals)")
        print("   • python utils/portfolio_sizer.py           (size a trade ticket)")
        print("   • python utils/csv_to_md.py data/live_alpha_us.csv  (markdown report)")
    else:
        print("\n⚠️  No universes scanned successfully.")


# =============================================================================
# CLI entry point
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-of-day ShockArb scanner — score today's tape and export CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--universe", "-u", nargs="+", default=["us", "global"],
        help="Universe(s) to scan (default: us global)",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override data directory (default: ./data or $SHOCK_ARB_DATA_DIR)",
    )

    parser.add_argument("--from-open", "-O", action="store_true",
                    help="Use today's session open as denominator (pure intraday)")
    
    parser.add_argument("--use-prior-close", "-p", action="store_true",
                    help="Force daily close-to-close returns")
    
    args = parser.parse_args()

    exec_cfg = ExecutionConfig(data_dir=args.data_dir, log_to_file=False)
    run_scanner(args.universe, exec_cfg,
            force_daily=args.use_prior_close,
            from_open=args.from_open)


if __name__ == "__main__":
    main()
