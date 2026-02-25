@echo off
echo ============================================================
echo   3D DVC Dataset Generator
echo ============================================================
echo.
echo   1. Generate dataset
echo   2. Diagnose TIF files
echo   3. Custom config path
echo   4. Exit
echo.
set /p choice="Select [1-4]: "

if "%choice%"=="1" goto generate
if "%choice%"=="2" goto diagnose
if "%choice%"=="3" goto custom
if "%choice%"=="4" goto end

echo Invalid choice. Please enter 1, 2, 3, or 4.
goto end

:generate
echo.
echo Available configs:
echo.
setlocal enabledelayedexpansion
set count=0
for %%f in (configs\*.yaml) do (
    set /a count+=1
    set "config_!count!=%%f"
    echo   !count!. %%~nf
)
if !count!==0 (
    echo No config files found in configs\
    goto end
)
echo.
set /p cfg_choice="Select config [1-!count!]: "
set "selected=!config_%cfg_choice%!"
if "!selected!"=="" (
    echo Invalid selection.
    goto end
)
echo.
echo Running: python -m dvc_generator.cli --config "!selected!"
python -m dvc_generator.cli --config "!selected!"
endlocal
goto end

:diagnose
echo.
set /p tif_dir="Enter TIF directory path (e.g. data_tif/my_experiment): "
python -m dvc_generator.analysis.diagnostics --tif_dir "%tif_dir%"
goto end

:custom
echo.
set /p config_path="Enter config file path (e.g. configs/synthetic_128_v1.yaml): "
python -m dvc_generator.cli --config "%config_path%"
goto end

:end
echo.
pause
