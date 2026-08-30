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

REM Capture the command, then collect any remaining args (e.g. --rvol,
REM --intraday, --no-rvol) so they can be forwarded to `stockfit report`.
set COMMAND=%1
set EXTRA_ARGS=
:collect_extra
shift /1
if "%1"=="" goto :after_collect
set EXTRA_ARGS=%EXTRA_ARGS% %1
goto :collect_extra
:after_collect

if "%COMMAND%"=="" goto :help
goto :%COMMAND% 2>nul || (echo Unknown command: %COMMAND% && goto :help)


REM ── SCORE ───────────────────────────────────────────────────
:score
echo [ShockArb] Scoring against ukraine_shock regime...
python -m shockarb score --regime ukraine_shock
goto :end

REM ── MARKET_REPORT ───────────────────────────────────────────
:market_report
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating report (rules-based)...
python -m marketfit report
echo Done. Output: reports\market_report.md
goto :end

REM ── MARKET_REPORT_LLM (alias — LLM now default) ────────────
:market_report_llm
echo [MarketFit] Refreshing market snapshot...
python utils\market_data.py
echo [MarketFit] Generating LLM-enhanced timestamped report...
python -m marketfit report --timestamp
echo Done. Timestamped report written to reports\
goto :end

REM ── MARKET_INTRADAY ─────────────────────────────────────────
:market_intraday
echo [MarketFit] Fetching live intraday prices...
python utils\market_data.py --intraday
echo [MarketFit] Generating intraday report...
python -m marketfit report --timestamp
echo Done. Output: reports\market_report_intraday*.md
goto :end

REM ── SHOCKARB_SCORE  (canonical training-corpus run) ─────────
:shockarb_score
echo [ShockArb] Full scoring run (score + marketfit)...
python -m shockarb score --regime ukraine_shock
python -m marketfit report --timestamp
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
python -m stockfit report%EXTRA_ARGS%
echo Done. Output: reports\stock_report.md
goto :end

REM ── STOCK_REPORT_LLM (alias — LLM now default) ──────────────
:stock_report_llm
echo [StockFit] Generating LLM stock report (timestamped)...
python -m stockfit report --timestamp%EXTRA_ARGS%
echo Done. Timestamped stock report written to reports\
goto :end

REM ── IRAN_REPORT (score + stock report under iran_shock regime) ─
:iran_report
echo [ShockArb] Scoring against iran_shock regime...
python -m shockarb score --regime iran_shock --out data\live_alpha_iran.csv
echo [StockFit] Generating stock report (iran_shock)...
python -m stockfit report --scores data\live_alpha_iran.csv --timestamp --save-verdicts --reports-dir reports\iran_shock%EXTRA_ARGS%
echo Done. Output: reports\
goto :end

REM ── DUAL_EOD (both regimes scored + compared) ────────────────
:dual_eod
echo [DUAL EOD] Starting dual-regime end-of-day workflow...
set /a DUAL_EOD_FAILURES=0
set DUAL_EOD_VERDICTS_OK=1

echo Step 1/7: Ukraine shock score
python -m shockarb score --regime ukraine_shock --out data\live_alpha_us.csv
if errorlevel 1 (echo [DUAL EOD] Step 1/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1)

echo Step 2/7: Iran shock score
python -m shockarb score --regime iran_shock --out data\live_alpha_iran.csv
if errorlevel 1 (echo [DUAL EOD] Step 2/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1)

echo Step 3/7: News + fundamentals
python utils\news_scanner.py
if errorlevel 1 (echo [DUAL EOD] Step 3/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1)

echo Step 4/7: Market snapshot
python utils\market_data.py
if errorlevel 1 (echo [DUAL EOD] Step 4/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1)

echo Step 5/7: MarketFit report (timestamped)
python -m marketfit report --timestamp
if errorlevel 1 (echo [DUAL EOD] Step 5/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1)

echo Step 6/7: Ukraine stock report (timestamped + verdicts)
python -m stockfit report --timestamp --save-verdicts%EXTRA_ARGS%
if errorlevel 1 (echo [DUAL EOD] Step 6/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1 & set DUAL_EOD_VERDICTS_OK=0)

echo Step 7/7: Iran stock report (timestamped + verdicts)
python -m stockfit report --scores data\live_alpha_iran.csv --timestamp --save-verdicts --reports-dir reports\iran_shock%EXTRA_ARGS%
if errorlevel 1 (echo [DUAL EOD] Step 7/7 FAILED - see traceback above & set /a DUAL_EOD_FAILURES+=1 & set DUAL_EOD_VERDICTS_OK=0)

if %DUAL_EOD_FAILURES% GTR 0 (
    echo [DUAL EOD] %DUAL_EOD_FAILURES% of 7 steps FAILED. Reports are missing or incomplete — see errors above.
    if "%DUAL_EOD_VERDICTS_OK%"=="0" (
        echo [DUAL EOD] Skipping compare — one or both verdicts CSVs were not written this run.
        exit /b 1
    )
)

echo [DUAL EOD] Reports written. Running compare...
call %~f0 compare_latest
if errorlevel 1 (
    echo [DUAL EOD] compare_latest failed.
    exit /b 1
)
if %DUAL_EOD_FAILURES% GTR 0 exit /b 1
goto :end

REM ── COMPARE_LATEST (auto-discover two newest verdicts CSVs) ──
REM 2026-08-30 (1st fix): was `dir /b` (non-recursive), which only ever sees
REM verdicts CSVs written directly into reports\ -- never the ones dual_eod's
REM own iran-regime step writes into reports\iran_shock\ (or eod5's
REM reports\global\), so "compare" reported "only one verdicts CSV found"
REM on every single dual_eod run.
REM
REM 2026-08-30 (2nd fix, same day): switching to `dir /s` fixed that error,
REM but `dir /s /o-d` does NOT produce one globally date-sorted list across
REM subfolders -- it sorts each directory's matches separately, then lists
REM directory blocks in traversal order. Since reports\ (containing the
REM Ukraine-regime CSVs) is traversed before reports\iran_shock\, F1/F2 kept
REM landing on the two most recent *Ukraine* runs instead of Ukraine vs.
REM Iran from the same run -- which defeats the point of "BOTH regimes
REM scored + compared". Replaced with a PowerShell one-liner, which sorts
REM correctly across the whole recursive result set in a single pass.
:compare_latest
echo [Compare] Finding two most recent verdicts CSVs...
set "F1="
set "F2="
for /f "delims=" %%A in ('powershell -NoProfile -Command "Get-ChildItem -Path reports -Filter stock_report_*_verdicts.csv -Recurse ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 2 -ExpandProperty FullName" 2^>nul') do (
    if not defined F1 (set "F1=%%A") else if not defined F2 (set "F2=%%A")
)
if not defined F1 (echo No verdicts CSVs found in reports\. Run with --save-verdicts first. && goto :end)
if not defined F2 (echo Only one verdicts CSV found — need two to compare. && goto :end)
echo Comparing: %F1%
echo      with: %F2%
python -m shockarb compare-reports "%F1%" "%F2%" --out reports\compare_latest.md
echo Done. Output: reports\compare_latest.md
goto :end

REM ── EOD (full end-of-day workflow, single regime) ────────────
:eod
echo [EOD] Starting full end-of-day workflow...
echo Step 1/5: ShockArb score
python -m shockarb score --regime ukraine_shock
echo Step 2/5: News + fundamentals
python utils\news_scanner.py
echo Step 3/5: Market snapshot
python utils\market_data.py
echo Step 4/5: MarketFit report (timestamped, LLM default)
python -m marketfit report --timestamp
echo Step 5/5: StockFit report (timestamped, LLM default)
python -m stockfit report --timestamp --save-verdicts%EXTRA_ARGS%
echo [EOD] Complete. Check reports\ for all outputs.
goto :end

REM ── GLOBAL_EOD (global ADR universe, global_ukraine_shock regime) ──────────
:global_eod
echo [GLOBAL EOD] Starting global ADR end-of-day workflow...
echo Step 1/5: Global universe score (global_ukraine_shock)
python -m shockarb score --regime global_ukraine_shock --out data\live_alpha_global.csv
echo Step 2/5: News + fundamentals
python utils\news_scanner.py
echo Step 3/5: Market snapshot
python utils\market_data.py
echo Step 4/5: MarketFit report (timestamped)
python -m marketfit report --timestamp
echo Step 5/5: Global stock report (timestamped + verdicts)
python -m stockfit report --scores data\live_alpha_global.csv --timestamp --save-verdicts --reports-dir reports\global%EXTRA_ARGS%
echo [GLOBAL EOD] Complete. Check reports\global\ for stock report.
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
echo  NOTE: LLM is ON by default for all report commands.
echo        Use --no-llm to get a fast rules-based report only.
echo.
echo  Commands:
echo    score              Run shockarb score (sticky regime)
echo    market_report      Refresh snapshot + market report (LLM default)
echo    market_report_llm  Alias for market_report --timestamp
echo    market_intraday    Live intraday prices + market report (timestamped)
echo    stock_report       Stock opportunity report (LLM default)
echo    stock_report_llm   Alias for stock_report --timestamp
echo    iran_report        Score iran_shock + stock report (timestamped + verdicts)
echo    dual_eod           BOTH regimes scored + compared (recommended daily)
echo    compare_latest     Auto-compare two most recent verdicts CSVs in reports\
echo    shockarb_score     score + market report combined
echo    news               Fetch headlines + fundamentals
echo    portfolio          Size positions from live_alpha_us.csv
echo    global_eod         Global ADR EOD: score global_ukraine_shock + full report chain
echo    eod / full         Full EOD: score + news + market + marketfit + stockfit
echo    daily_scan         Run daily_scanner.py
echo    build              Build/rebuild factor model
echo.
echo  Recommended daily workflow (dual-regime):
echo    scripts\shockarb_workflows.bat dual_eod
echo.
echo  Single-regime EOD:
echo    scripts\shockarb_workflows.bat eod
echo.
echo  Compare latest two reports manually:
echo    scripts\shockarb_workflows.bat compare_latest
echo.
echo  Rules-based only (no LLM, fast):
echo    python -m stockfit report --no-llm
echo    python -m marketfit report --no-llm
echo.
echo  Extra args after the command are forwarded to `stockfit report`
echo  (stock_report, stock_report_llm, iran_report, eod/full, dual_eod), e.g.:
echo    scripts\shockarb_workflows.bat eod --rvol
echo    scripts\shockarb_workflows.bat dual_eod --rvol --intraday
echo.

:end
