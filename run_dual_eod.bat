@echo off
setlocal

:: ============================================================================
:: ShockArb dual_eod launcher
::
:: Replaces the manual routine (open Anaconda -> activate "manager" ->
:: cd shockarb_lab -> scripts\shockarb_workflows dual_eod) with one
:: double-click. Also safe to trigger from Windows Task Scheduler, since it
:: does not depend on an already-open Anaconda Prompt.
::
:: If the "conda activate" step fails, your Anaconda install path differs
:: from the default guessed below. Open your usual Anaconda Prompt and run:
::     conda info --base
:: then replace CONDA_ACTIVATE's directory with whatever that prints.
:: ============================================================================

set "CONDA_ACTIVATE=C:\Users\nlgun\miniconda3\Scripts\activate.bat"
set "PROJECT_DIR=%~dp0"
set "LOGFILE=%PROJECT_DIR%logs\dual_eod_run.log"

if not exist "%PROJECT_DIR%logs" mkdir "%PROJECT_DIR%logs"

echo.>> "%LOGFILE%"
echo ==== Run started %date% %time% ==== >> "%LOGFILE%"

call "%CONDA_ACTIVATE%" manager >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo ERROR: conda activate failed - see CONDA_ACTIVATE path in this script >> "%LOGFILE%"
    echo Conda activate failed. Check logs\dual_eod_run.log for details.
    exit /b 1
)

cd /d "%PROJECT_DIR%"
call scripts\shockarb_workflows.bat dual_eod >> "%LOGFILE%" 2>&1

echo ==== Run finished %date% %time% ==== >> "%LOGFILE%"
echo Done. Full output logged to logs\dual_eod_run.log

endlocal
