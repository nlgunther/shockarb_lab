"""
Throwaway diagnostic: trace date alignment between

  - etf_returns / stock_returns = prices_to_returns(prices).iloc[-1]
  - today_close / yesterday_close = pd.concat([etf_prices, stock_prices], axis=1).ffill().iloc[-1]/[-2]

for score_universe('us_iran', force_daily=True), focused on KLAC, LRCX, AMAT
(the "outperformed factors" names with suspicious +9-13% one-day moves) plus
TLT as an ETF reference.

Does NOT modify pipeline.py. Delete once the bug is diagnosed/fixed.

Run:
    python diag_date_alignment.py [regime_name]
"""

import sys
import pandas as pd

import shockarb.pipeline as pipeline
from shockarb.config import ExecutionConfig
from shockarb.regimes import get_regime

REGIME_NAME = sys.argv[1] if len(sys.argv) > 1 else "iran_shock"
WATCH_TICKERS = ["KLAC", "LRCX", "AMAT", "TLT", "TXN"]

regime = get_regime(REGIME_NAME)
universe = regime.universe
exec_cfg = ExecutionConfig(log_to_file=False)

model_path = pipeline.find_latest_model(universe.name, exec_cfg, regime=regime.name)
if not model_path:
    sys.exit(f"No model found for regime '{regime.name}' / universe '{universe.name}'. Run 'build' first.")
model = pipeline.load_model(model_path)
print(f"Loaded model: {model_path}")

# --- capture prices_to_returns(prices) -> returns, in call order ---
_orig_p2r = pipeline.prices_to_returns
p2r_calls = []  # list of (input_prices, output_returns)


def traced_p2r(prices, *a, **kw):
    out = _orig_p2r(prices, *a, **kw)
    p2r_calls.append((prices, out))
    return out


pipeline.prices_to_returns = traced_p2r

# --- capture the pd.concat([etf_prices, stock_prices], axis=1) call ---
_orig_concat = pipeline.pd.concat
all_prices_holder = {}


def traced_concat(objs, *a, **kw):
    objs_list = list(objs)
    axis = kw.get("axis", a[0] if a else 0)
    result = _orig_concat(objs_list, *a, **kw)
    # the all_prices concat: axis=1, exactly 2 DataFrame operands, result is 2D
    if (
        axis == 1
        and len(objs_list) == 2
        and all(isinstance(o, pd.DataFrame) for o in objs_list)
        and "all_prices" not in all_prices_holder
        and getattr(result, "ndim", None) == 2
        and result.shape[1] >= 70  # 19 ETFs + 66 stocks, roughly
    ):
        all_prices_holder["all_prices"] = result
    return result


pipeline.pd.concat = traced_concat

print(f"\nRunning score_universe('{universe.name}', regime='{regime.name}', force_daily=True) ...")
try:
    scores, prov = pipeline.score_universe(universe, model, exec_cfg, force_daily=True)
    print("\nscore_universe completed OK.\n")
finally:
    pipeline.pd.concat = _orig_concat
    pipeline.prices_to_returns = _orig_p2r

# --- p2r_calls[0] = etf (19 cols), p2r_calls[1] = stock (66 cols) ---
print(f"prices_to_returns() called {len(p2r_calls)} time(s).")
for i, (prices, returns) in enumerate(p2r_calls):
    label = "etf" if i == 0 else "stock" if i == 1 else f"call{i}"
    print(f"\n[{label}] prices: shape={prices.shape}, "
          f"index range = {prices.index[0].date()} .. {prices.index[-1].date()}")
    print(f"[{label}] returns: shape={returns.shape}, "
          f"index range = {returns.index[0].date()} .. {returns.index[-1].date()}, "
          f".iloc[-1] date = {returns.index[-1].date()}")

all_prices = all_prices_holder.get("all_prices")
if all_prices is not None:
    print(f"\n[all_prices] shape={all_prices.shape}, "
          f"index range = {all_prices.index[0].date()} .. {all_prices.index[-1].date()}")
    print(f"  today_date     (iloc[-1]) = {all_prices.index[-1].date()}")
    print(f"  yesterday_date (iloc[-2]) = {all_prices.index[-2].date()}")
else:
    print("\n[all_prices] NOT CAPTURED -- adjust the shape filter in traced_concat.")

# --- per-ticker detail ---
for ticker in WATCH_TICKERS:
    print(f"\n=== {ticker} ===")
    for i, (prices, returns) in enumerate(p2r_calls):
        label = "etf" if i == 0 else "stock" if i == 1 else f"call{i}"
        if ticker not in prices.columns:
            continue
        px = prices[ticker].dropna()
        print(f"  [{label}] raw price history (last 3): "
              + ", ".join(f"{d.date()}={v:.4f}" for d, v in px.tail(3).items()))
        if ticker in returns.index:
            ret_val = returns[ticker]
            ret_date = returns.index[-1].date()  # .iloc[-1] is a row -> Series indexed by ticker
            print(f"  [{label}] prices_to_returns(...).iloc[-1][{ticker}] = {ret_val:.6f} "
                  f"(row date = {ret_date})")

    if all_prices is not None and ticker in all_prices.columns:
        ffilled = all_prices.ffill()
        today_v = ffilled[ticker].iloc[-1]
        yest_v = ffilled[ticker].iloc[-2]
        today_d = ffilled.index[-1].date()
        yest_d = ffilled.index[-2].date()
        manual_ret = today_v / yest_v - 1 if yest_v else float("nan")
        print(f"  [all_prices] today  ({today_d}) = {today_v:.6f}")
        print(f"  [all_prices] yesterday ({yest_d}) = {yest_v:.6f}")
        print(f"  [all_prices] manual today/yesterday - 1 = {manual_ret:.6f}")
