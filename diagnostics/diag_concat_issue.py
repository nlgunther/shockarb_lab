"""
Throwaway diagnostic for the `pd.concat` AttributeError seen in
`pipeline.score_universe()` around line 816:

    AttributeError: 'Series' object has no attribute 'columns'

Does NOT modify pipeline.py. It monkeypatches `prices_to_returns` and
`pd.concat` (as seen from inside the pipeline module) purely to print
diagnostics, then calls score_universe(force_daily=True) to force the
Path A (daily) branch where the failure occurs.

Run from the project root (same env/PYTHONPATH as shockarb_workflows.bat):

    python diag_concat_issue.py [regime_name]

Default regime is "iran_shock". Delete this file once the bug is
diagnosed/fixed.
"""

import sys
import traceback

import pandas as pd
import numpy as np

print(f"pandas {pd.__version__}, numpy {np.__version__}, python {sys.version}")

import shockarb.pipeline as pipeline
from shockarb.config import ExecutionConfig
from shockarb.regimes import get_regime

REGIME_NAME = sys.argv[1] if len(sys.argv) > 1 else "iran_shock"
regime = get_regime(REGIME_NAME)
universe = regime.universe

exec_cfg = ExecutionConfig(log_to_file=False)

model_path = pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
if not model_path:
    sys.exit(f"No model found for regime '{regime.name}' / universe '{universe.name}'. Run 'build' first.")

model = pipeline.load_model(model_path)
print(f"Loaded model: {model_path}")


def _describe(label, obj):
    info = [f"type={type(obj).__name__}"]
    if hasattr(obj, "ndim"):
        info.append(f"ndim={obj.ndim}")
    if hasattr(obj, "shape"):
        info.append(f"shape={obj.shape}")
    if hasattr(obj, "name"):
        info.append(f"name={obj.name!r}")
    if hasattr(obj, "index"):
        info.append(f"index_type={type(obj.index).__name__}")
    if hasattr(obj, "columns"):
        cols = list(obj.columns)
        info.append(f"columns={cols[:6]}{'...' if len(cols) > 6 else ''}")
    if hasattr(obj, "dtype"):
        info.append(f"dtype={obj.dtype}")
    elif hasattr(obj, "dtypes"):
        info.append("dtypes=" + obj.dtypes.to_dict().__repr__())
    print(f"  {label}: " + ", ".join(info))


# --- wrap prices_to_returns to inspect its output ---
_orig_p2r = pipeline.prices_to_returns


def traced_p2r(prices, *a, **kw):
    out = _orig_p2r(prices, *a, **kw)
    print("\n[prices_to_returns]")
    _describe("input ", prices)
    _describe("output", out)
    if hasattr(out, "iloc") and len(out):
        row = out.iloc[-1]
        _describe("output.iloc[-1]", row)
    return out


pipeline.prices_to_returns = traced_p2r

# --- wrap pd.concat as seen by the pipeline module ---
_orig_concat = pipeline.pd.concat


def traced_concat(objs, *a, **kw):
    objs_list = list(objs)
    axis = kw.get("axis", a[0] if a else 0)
    print(f"\n[pd.concat] n={len(objs_list)} axis={axis}")
    for i, o in enumerate(objs_list):
        _describe(f"obj[{i}]", o)
    try:
        return _orig_concat(objs_list, *a, **kw)
    except Exception:
        print("\n[pd.concat] FAILED -- traceback:")
        traceback.print_exc()
        raise


pipeline.pd.concat = traced_concat

print(f"\nRunning score_universe('{universe.name}', regime='{regime.name}', force_daily=True) ...")
try:
    scores, prov = pipeline.score_universe(universe, model, exec_cfg, force_daily=True)
    print("\nOK -- score_universe completed without error.")
    print(scores.head())
except Exception:
    print("\nscore_universe raised:")
    traceback.print_exc()
finally:
    pipeline.pd.concat = _orig_concat
    pipeline.prices_to_returns = _orig_p2r
