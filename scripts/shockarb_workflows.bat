@echo off
REM ============================================================
REM  ShockArb Workflow Commands
REM  Run from project root: scripts\shockarb_workflows.bat <CMD>
REM  Usage: shockarb_workflows.bat score
REM ============================================================

if "%1"=="" goto :help
goto :%1 2>nul || (echo Unknown command: %1 && goto :help)


REM ── SCORE ───────────────────────────────────────────────────
:score
echo [ShockArb] Scoring against sticky regime...
shockarb score
goto :end

REM ── MARKET_REPORT ───────────────────────────────────────────
:market_report
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating report (rules-based)...
cd utils && python -m marketfit report && cd ..
echo Done. Output: data\market_report.md
goto :end

REM ── MARKET_REPORT_LLM ───────────────────────────────────────
:market_report_llm
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating LLM-enhanced timestamped report...
cd utils && python -m marketfit report --llm --timestamp && cd ..
echo Done. Timestamped report written to data\
goto :end

REM ── SHOCKARB_SCORE  (canonical training-corpus run) ─────────
:shockarb_score
echo [ShockArb] Full scoring run (score + marketfit)...
shockarb score
cd utils && python -m marketfit report --llm --timestamp && cd ..
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

REM ── EOD (full end-of-day workflow) ───────────────────────────
:eod
echo [EOD] Starting full end-of-day workflow...
echo Step 1/4: ShockArb score
shockarb score
echo Step 2/4: News + fundamentals
python utils\news_scanner.py
echo Step 3/4: Market snapshot
python utils\market_data.py
echo Step 4/4: MarketFit LLM report (timestamped)
cd utils && python -m marketfit report --llm --timestamp && cd ..
echo [EOD] Complete. Check data\ for all outputs.
goto :end

REM ── DAILY_SCAN ───────────────────────────────────────────────
:daily_scan
echo [Daily] Running daily scanner...
python utils\daily_scanner.py
goto :end

REM ── BUILD ────────────────────────────────────────────────────
:build
echo [Build] Building factor model for sticky regime...
shockarb build
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
echo    market_report_llm  Refresh snapshot + LLM report (timestamped corpus)
echo    shockarb_score     score + market_report_llm combined
echo    news               Fetch headlines + fundamentals
echo    portfolio          Size positions from live_alpha_us.csv
echo    eod                Full end-of-day: score + news + market + LLM report
echo    daily_scan         Run daily_scanner.py
echo    build              Build/rebuild factor model
echo.
echo  Typical training-corpus run:
echo    scripts\shockarb_workflows.bat shockarb_score
echo.
echo  Or inline from project root:
echo    cd utils ^&^& python -m marketfit report --llm --timestamp ^&^& cd ..
echo.

:end
