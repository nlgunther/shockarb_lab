"""
Utility script to convert a raw ShockArb alpha sheet (CSV) into a 
human-readable Markdown report for easy sharing, rendering, or documentation.
"""

import argparse
import os
import pandas as pd
from loguru import logger

# Safely import the resolver from the new package structure
try:
    from shockarb.names import TickerReferenceResolver
except ImportError:
    try:
        # Fallback in case you put names.py in the root folder instead
        from names import TickerReferenceResolver
    except ImportError:
        TickerReferenceResolver = None
        logger.warning("Could not find 'names.py'. Name resolution will be skipped.")
        logger.warning("Please copy 'names.py' into your 'shockarb/' directory.")


def format_percentage(val: float) -> str:
    """Formats a decimal as a clean percentage with explicit positive/negative signs."""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:+.2f}%"


def generate_markdown_report(csv_path: str, output_path: str = None, resolve_names: bool = False, ref_dir: str = "./data"):
    """
    Reads the live alpha CSV and translates the mathematical output into 
    a highly readable Markdown table with contextual headers.
    """
    logger.info(f"Loading alpha sheet from {csv_path}...")
    try:
        # Load the CSV, setting the first column (tickers) as the index
        df = pd.read_csv(csv_path, index_col=0)
    except Exception as e:
        logger.error(f"Failed to read {csv_path}. Ensure the file exists: {e}")
        return

    # Ensure the dataframe is sorted by the primary conviction metric
    if 'confidence_delta' in df.columns:
        df = df.sort_values(by='confidence_delta', ascending=False)

    # Instantiate a clean DataFrame strictly for Markdown formatting
    md_df = pd.DataFrame(index=df.index)
    
    # --- OPTIONAL REFERENCE INJECTION ---
    if resolve_names:
        if TickerReferenceResolver is None:
            logger.error("Cannot resolve names: TickerReferenceResolver is not available.")
        else:
            logger.info(f"Name resolution requested. Checking local exchange references in {ref_dir}...")
            
            filepaths = [os.path.join(ref_dir, f'{k}.csv')
                         for k in 'nasadaq_3105 nasdaq_1668526380140 nyse_3105 nyse_1668526574444'.split()]
            
            resolver = TickerReferenceResolver(
                file_paths=filepaths,
                cache_path="./data/ticker_reference_cache.json"
            )
            ref_map = resolver.get_reference(df.index.tolist())
            
            md_df['Company Name'] = df.index.map(lambda x: ref_map.get(x, {}).get("Name", x))
            md_df['Industry'] = df.index.map(lambda x: ref_map.get(x, {}).get("Industry", "Unknown"))
    
    # Map out the exact columns that require percentage formatting
    pct_cols = [
        'actual_return', 'expected_return', 'expected_rel', 'expected_abs', 
        'delta', 'delta_rel', 'delta_abs', 'residual_vol', 'confidence_delta'
    ]
    
    for col in pct_cols:
        if col in df.columns:
            md_df[col] = df[col].apply(format_percentage)
            
    # Format R-squared specifically (strictly 3 decimals, no percentage sign)
    if 'r_squared' in df.columns:
        md_df['r_squared'] = df['r_squared'].apply(lambda x: f"{x:.3f}")

    # Rename mathematical column headers to human-friendly titles
    rename_map = {
        'actual_return': 'Actual Return',
        'expected_return': 'Expected Return',
        'expected_rel': 'Expected (Relative)',
        'expected_abs': 'Expected (Absolute)',
        'delta': 'Delta',
        'delta_rel': 'Delta (Relative)',
        'delta_abs': 'Delta (Absolute)',
        'r_squared': 'R-Squared',
        'residual_vol': 'Residual Vol',
        'confidence_delta': 'Confidence Delta'
    }
    md_df = md_df.rename(columns=rename_map)
    md_df.index.name = 'Ticker'

    # Extract the original filename to document the data source
    filename = os.path.basename(csv_path)
    
    filename_lower = filename.lower()
    if "_global." in filename_lower or "_global_" in filename_lower:
        universe_label = "GLOBAL"
    elif "_us." in filename_lower or "_us_" in filename_lower:
        universe_label = "US"
    else:
        universe_label = "UNKNOWN"
    
    # Construct the Markdown document architecture
    md_lines = [
        f"# ⚡ ShockArb Alpha Report ({universe_label})",
        f"**Source File:** `{filename}`\n",
        "> **How to read this report:**",
        "> * **Delta:** The pure arbitrage signal. Positive means the stock dropped *more* than the macro factors justified.",
        "> * **R-Squared:** The model's historical accuracy for this stock. Higher means the baseline factor relationship is more reliable.",
        "> * **Confidence Delta:** The ultimate conviction-weighted signal (`Delta` × `R-Squared`).\n",
        "### Actionable Targets",
        md_df.to_markdown()
    ]
    
    md_content = "\n".join(md_lines)
    
    # Default to saving the Markdown file in the same directory with a .md extension
    if not output_path:
        output_path = csv_path.replace('.csv', '.md')
        
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        logger.success(f"Markdown report generated successfully: {output_path}")
    except Exception as e:
        logger.error(f"Failed to write Markdown file to {output_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ShockArb Alpha CSV to a Markdown Report")
    parser.add_argument("csv_file", type=str, help="Absolute or relative path to the live alpha CSV file")
    parser.add_argument("--out", type=str, help="Optional specific output path for the Markdown file", default=None)
    parser.add_argument("--names", action="store_true", help="Look up and include full company names and industries from local CSVs")
    parser.add_argument("--ref-dir", type=str, default="./data/Global-Stock-Symbols", help="Path to the directory containing reference CSVs")
    
    args = parser.parse_args()
    
    generate_markdown_report(args.csv_file, args.out, args.names, args.ref_dir)