@echo off
REM ============================================================
REM  ShockArb Workflow Commands
REM  Can be run from any directory.
REM  Usage: scripts\shockarb_workflows.bat score
REM ============================================================

REM Always run from the project root (one level above scripts\)
cd /d "%~dp0.."

REM Add utils\ to PYTHONPATH so marketfit and stockfit packages are importable
set PYTHONPATH=%cd%\utils;%PYTHONPATH%

if "%1"=="" goto :help
goto :%1 2>nul || (echo Unknown command: %1 && goto :help)


REM ── SCORE ───────────────────────────────────────────────────
:score
echo [ShockArb] Scoring against sticky regime...
python -m shockarb score
goto :end

REM ── MARKET_REPORT ───────────────────────────────────────────
:market_report
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating report (rules-based)...
python -m marketfit report
echo Done. Output: reports\market_report.md
goto :end

REM ── MARKET_REPORT_LLM ───────────────────────────────────────
:market_report_llm
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating LLM-enhanced timestamped report...
python -m marketfit report --llm --timestamp
echo Done. Timestamped report written to reports\
goto :end

REM ── MARKET_INTRADAY ─────────────────────────────────────────
:market_intraday
echo [MarketFit] Fetching live intraday prices...
python utils\market_data.py --intraday
echo [MarketFit] Generating intraday report...
python -m marketfit report --llm --timestamp
echo Done. Output: reports\market_report_intraday*.md
goto :end

REM ── SHOCKARB_SCORE  (canonical training-corpus run) ─────────
:shockarb_score
echo [ShockArb] Full scoring run (score + marketfit)...
python -m shockarb score
python -m marketfit report --llm --timestamp
goto :end

REM ── NEWS ────────────────────────────────────────────────────
:news
echo [News] Fetching headlines and fundamentals...
python utils\news_scanner.py
echo Done. Output: data\news.txt, data\fundamentals.csv
goto :end

REM ── PORTFOLIO ────────────────────────────────────────────────
:portfolio
echo [Portfolio] Sizing positions from live_alpha_us.csv...
python utils\portfolio_sizer.py
echo Done. Output: data\portfolio_sizer.csv
goto :end

REM ── STOCK_REPORT ─────────────────────────────────────────────
:stock_report
echo [StockFit] Generating stock opportunity report (rules-based)...
python -m stockfit report
echo Done. Output: reports\stock_report.md
goto :end

REM ── STOCK_REPORT_LLM ─────────────────────────────────────────
:stock_report_llm
echo [StockFit] Generating LLM-enhanced stock report (timestamped)...
python -m stockfit report --llm --timestamp
echo Done. Timestamped stock report written to reports\
goto :end

REM ── IRAN_REPORT (score + stock report under iran_shock regime) ─
:iran_report
echo [ShockArb] Scoring against iran_shock regime...
if not exist data\iran_shock mkdir data\iran_shock
if not exist reports\iran_shock mkdir reports\iran_shock
python -m shockarb score --regime iran_shock --out data\iran_shock\live_alpha_us.csv
echo [News] Refreshing fundamentals for iran_shock candidates...
python utils\news_scanner.py --csv data\iran_shock\live_alpha_us.csv --top 20
echo [StockFit] Generating LLM-enhanced stock report (iran_shock)...
python -m stockfit report --scores data\iran_shock\live_alpha_us.csv --reports-dir reports\iran_shock --llm --timestamp
echo Done. Output: reports\iran_shock\
goto :end

REM ── EOD (full end-of-day workflow) ───────────────────────────
:eod
echo [EOD] Starting full end-of-day workflow...
echo Step 1/5: ShockArb score
python -m shockarb score
echo Step 2/5: News + fundamentals
python utils\news_scanner.py
echo Step 3/5: Market snapshot
python utils\market_data.py
echo Step 4/5: MarketFit LLM report (timestamped)
python -m marketfit report --llm --timestamp
echo Step 5/5: StockFit LLM report (timestamped)
python -m stockfit report --llm --timestamp
echo [EOD] Complete. Check data\ for all outputs.
goto :end

REM ── FULL (EOD alias) ─────────────────────────────────────────
:full
goto :eod

REM ── DAILY_SCAN ───────────────────────────────────────────────
:daily_scan
echo [Daily] Running daily scanner...
python utils\daily_scanner.py
goto :end

REM ── BUILD ────────────────────────────────────────────────────
:build
echo [Build] Building factor model for sticky regime...
python -m shockarb build
goto :end

REM ── HELP ─────────────────────────────────────────────────────
:help
echo.
echo  ShockArb Workflow Commands
echo  ==========================
echo  scripts\shockarb_workflows.bat ^<command^>
echo.
echo  Commands:
echo    score              Run shockarb score (sticky regime)
echo    market_report      Refresh snapshot + rules-based marketfit report
echo    market_report_llm  Refresh snapshot + LLM market report (timestamped)
echo    market_intraday    Live intraday prices + LLM report (timestamped)
echo    stock_report       Rules-based stock opportunity report
echo    stock_report_llm   LLM-enhanced stock report (timestamped)
echo    iran_report        Score + stock report under iran_shock regime (separate folder)
echo    shockarb_score     score + market_report_llm combined
echo    news               Fetch headlines + fundamentals
echo    portfolio          Size positions from live_alpha_us.csv
echo    eod / full         Full EOD: score + news + market + marketfit + stockfit
echo    daily_scan         Run daily_scanner.py
echo    build              Build/rebuild factor model
echo.
echo  Full EOD corpus run:
echo    scripts\shockarb_workflows.bat eod
echo.
echo  Stand-alone stock report with LLM:
echo    python -m stockfit report --llm --timestamp
echo.

:end
