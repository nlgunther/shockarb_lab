@echo off
REM ============================================================
REM  ShockArb Test Subset Runner
REM  Can be run from any directory.
REM  Usage: scripts\run_tests.bat <group> [extra pytest args...]
REM
REM  Lightweight, file-based grouping over tests\ (no custom
REM  pytest markers). Run "run_tests.bat help" for the full group list.
REM ============================================================

REM Always run from the project root (one level above scripts\)
cd /d "%~dp0.."

REM Add utils\ to PYTHONPATH so marketfit and stockfit packages are importable
set PYTHONPATH=%cd%\utils;%PYTHONPATH%

set GROUP=%1
if "%GROUP%"=="" set GROUP=help

REM Shift off the group name so %* below is "extra pytest args"
shift
set EXTRA=
:collect_extra
if "%1"=="" goto :run
set EXTRA=%EXTRA% %1
shift
goto :collect_extra

:run
if /i "%GROUP%"=="help"        goto :help
if /i "%GROUP%"=="all"         set FILES=tests& goto :pytest
if /i "%GROUP%"=="stockfit"    set FILES=tests\test_stockfit.py tests\test_stockfit_rvol.py& goto :pytest
if /i "%GROUP%"=="rvol"        set FILES=tests\test_stockfit_rvol.py& goto :pytest
if /i "%GROUP%"=="marketfit"   set FILES=tests\test_marketfit.py& goto :pytest
if /i "%GROUP%"=="report"      set FILES=tests\test_report.py& goto :pytest
if /i "%GROUP%"=="cli"         set FILES=tests\test_cli.py tests\test_cli_regime_additions.py tests\test_out_flag.py& goto :pytest
if /i "%GROUP%"=="engine"      set FILES=tests\test_engine.py tests\test_backtest.py& goto :pytest
if /i "%GROUP%"=="pipeline"    set FILES=tests\test_pipeline.py tests\test_pipeline_regime_additions.py tests\test_coordinator_phase1.py& goto :pytest
if /i "%GROUP%"=="regimes"     set FILES=tests\test_regimes.py tests\test_cli_regime_additions.py tests\test_pipeline_regime_additions.py& goto :pytest
if /i "%GROUP%"=="cache"       set FILES=tests\test_cache.py tests\test_score_history.py& goto :pytest
if /i "%GROUP%"=="config"      set FILES=tests\test_config.py& goto :pytest
if /i "%GROUP%"=="portfolio"   set FILES=tests\test_portfolio_sizer.py& goto :pytest
if /i "%GROUP%"=="fundamentals" set FILES=tests\test_fundamental_scanner.py& goto :pytest

echo Unknown group: %GROUP%
goto :help

:pytest
echo [run_tests] pytest %FILES% %EXTRA%
python -m pytest %FILES% %EXTRA%
goto :end

:help
echo.
echo  ShockArb Test Subset Runner
echo  ============================
echo  scripts\run_tests.bat ^<group^> [extra pytest args...]
echo.
echo  Groups:
echo    all            Full suite (tests\)
echo    stockfit       stockfit feature/rules/report/cli tests + RVOL tests
echo    rvol           RVOL feature tests only (test_stockfit_rvol.py)
echo    marketfit      marketfit tests
echo    report         shockarb report assembly tests
echo    cli            shockarb CLI + regime + --out flag tests
echo    engine         factor model engine + backtest tests
echo    pipeline       pipeline + coordinator + regime-pipeline tests
echo    regimes        regime definitions + regime-aware cli/pipeline tests
echo    cache          data cache + score history tests
echo    config         ExecutionConfig tests
echo    portfolio      portfolio sizer tests
echo    fundamentals   fundamental scanner tests
echo.
echo  Examples:
echo    scripts\run_tests.bat rvol
echo    scripts\run_tests.bat stockfit -v
echo    scripts\run_tests.bat all -k rvol
echo    scripts\run_tests.bat all
echo.

:end
