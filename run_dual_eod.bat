@echo off
setlocal

:: ============================================================================
:: ShockArb dual_eod launcher
::
:: Replaces the manual routine (open Anaconda -> activate "quant" ->
:: cd shockarb_lab -> scripts\shockarb_workflows dual_eod) with one
:: double-click. Also safe to trigger from Windows Task Scheduler, since it
:: does not depend on an already-open Anaconda Prompt.
::
:: Relies on conda.bat being resolvable via PATH (true whenever "conda.bat
:: activate <env>" works from an ordinary Command Prompt -- verified working
:: on this machine 2026-08-27). If it ever fails with "not recognized",
:: your conda Scripts\ directory has dropped off PATH; open your usual
:: Anaconda Prompt, run `where conda.bat`, and hardcode that full path here
:: instead of relying on PATH resolution.
::
:: 2026-08-30: env renamed from the old per-repo "shockarb" env to the
:: shared "quant" env (shockarb_lab, investing, statement_guard, etc. now
:: all live in one conda env -- see envs\README_conda_envs.md at the repo
:: root). This script's hardcoded "activate shockarb" was never updated
:: after that move, which is why it was failing with
:: "EnvironmentNameNotFound: Could not find conda environment: shockarb"
:: (the error message below was misleadingly blaming PATH resolution
:: instead -- that's fixed too).
::
:: 2026-08-30 (2nd fix, same day): steps 5-7 (marketfit, stockfit x2) were
:: crashing with UnicodeEncodeError ('charmap' codec can't encode ...) as
:: soon as they tried to print a report containing an emoji (e.g. U+1F4CA,
:: U+274C). Cause: Windows' legacy console codepage (cp1252) can't represent
:: those characters, and that's still what Python's stdout uses by default
:: even when output is redirected to a file, unless told otherwise. Fix:
:: force UTF-8 mode for this script's whole python/conda subtree below.
:: ============================================================================

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PROJECT_DIR=%~dp0"
set "LOGFILE=%PROJECT_DIR%logs\dual_eod_run.log"

if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"

echo.>> "%LOGFILE%"
echo ==== Run started %date% %time% ==== >> "%LOGFILE%"

call conda.bat activate quant >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ERROR: conda activate failed - either conda.bat is not on PATH, or the "quant" env does not exist yet ^(conda env create -f envs\quant.yml from the repo root^) >> "%LOGFILE%"
    echo Conda activate failed. Check logs\dual_eod_run.log for details.
    exit /b 1
)

cd /d "%PROJECT_DIR%"
call scripts\shockarb_workflows.bat dual_eod >> "%LOGFILE%" 2>&1
set "DUAL_EOD_RESULT=%errorlevel%"

echo ==== Run finished %date% %time% (exit %DUAL_EOD_RESULT%) ==== >> "%LOGFILE%"

if not "%DUAL_EOD_RESULT%"=="0" (
    echo FAILED - one or more steps errored. Check logs\dual_eod_run.log for details.
    exit /b 1
)
echo Done. Full output logged to logs\dual_eod_run.log

endlocal
