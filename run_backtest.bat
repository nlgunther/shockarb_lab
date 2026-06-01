@echo off
shockarb backtest --model data\ukraine_shock_us_20260528_143030.json --trailing-window 390 --return-type both --holding-periods 1 2 3 5 --top-n 5 --min-r-squared 0.50 --min-confidence 0.005
