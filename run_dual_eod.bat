@echo off
setlocal

:: ============================================================================
:: ShockArb dual_eod launcher
::
:: Replaces the manual routine (open Anaconda -> activate "shockarb" ->
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
:: ============================================================================

set "PROJECT_DIR=%~dp0"
set "LOGFILE=%PROJECT_DIR%logs\dual_eod_run.log"

if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"

echo.>> "%LOGFILE%"
echo ==== Run started %date% %time% ==== >> "%LOGFILE%"

call conda.bat activate shockarb >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ERROR: conda activate failed - conda.bat not found on PATH; see comment at top of this script >> "%LOGFILE%"
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
