@echo off
setlocal enabledelayedexpansion

title getModel

echo.
echo #######################################################################################
echo.
echo ----------           1. Batch Script to download Models from OpenVINO           ----------
echo ----------           2. Performs converion to IR Representation                 ----------
echo.
echo #######################################################################################
echo.

:: Setting up default values
set TARGET=CPU
set BUILD_FOLDER=%USERPROFILE%\Documents\
set ROOT_DIR=C:\Program Files (x86)\Intel\openvino_2021.3.394

set models_path=%BUILD_FOLDER%\openvino_models\models
set models_cache=%BUILD_FOLDER%\openvino_models\cache
set irs_path=%BUILD_FOLDER%\openvino_models\ir

:: Command Line Arguments Parsing
:input_arguments_loop
if not "%1"=="" (
    if "%1"=="--d" (
        set TARGET=%2
        shift
    )
    if "%1"=="--precision" (
        set TARGET_PRECISION=%2
        shift
    )
    if "%1"=="--model-name" (
        set MODEL_NAME=%2
        shift
    )
    shift
    goto :input_arguments_loop
)


:: Setting up OpenVINO Environment
if exist "%ROOT_DIR%\bin\setupvars.bat" (
    call "%ROOT_DIR%\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

:: Displaying Relevant Information
echo.
echo #################################################
echo.
echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%
echo.
echo Build Folder is set to %BUILD_FOLDER%\openvino_models
echo.
echo Target = %TARGET%
echo.
echo Model Name = %MODEL_NAME%
echo.
echo Precision = %TARGET_PRECISION%
echo.
echo #################################################
echo.


:: ####################################### WTF??? #########################################
:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Error^: Python is not installed. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
    goto error
)

:: Check if Python version is equal or higher 3.4
for /F "tokens=* USEBACKQ" %%F IN (`python --version 2^>^&1`) DO (
    set version=%%F
)
echo %var%

for /F "tokens=1,2,3 delims=. " %%a in ("%version%") do (
    set Major=%%b
    set Minor=%%c
)

if "%Major%" geq "3" (
    if "%Minor%" geq "5" (
        set python_ver=okay
    )
)

if not "%python_ver%"=="okay" (
    echo Unsupported Python version. Please install Python 3.5 ^(64-bit^) or higher from https://www.python.org/downloads/
    goto error
)

:: install yaml python modules required for downloader.py
pip3 install --user -r "%ROOT_DIR%\deployment_tools\open_model_zoo\tools\downloader\requirements.in"
if ERRORLEVEL 1 GOTO errorHandling


:: Setting variable to hold path that contains downloader script
set downloader_dir=%INTEL_OPENVINO_DIR%\deployment_tools\open_model_zoo\tools\downloader

for /F "tokens=* usebackq" %%d in (
    `python "%downloader_dir%\info_dumper.py" --name "%model_name%" ^|
        python -c "import sys, json; print(json.load(sys.stdin)[0]['subdirectory'])"`
) do (
    set model_dir=%%d
)
:: ###########################################################################################

set ir_dir=%irs_path%\%model_dir%\%target_precision%

echo.
echo #################################################
echo.
echo Download public %model_name% model
echo python "%downloader_dir%\downloader.py" --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
python "%downloader_dir%\downloader.py" --name "%model_name%" --output_dir "%models_path%" --cache_dir "%models_cache%"
echo %model_name% model downloading completed
echo.
echo #################################################
echo.

::echo.
::echo ###############^|^| Install Model Optimizer prerequisites ^|^|###############
::echo.
::CALL :delay 5
::cd /d "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\install_prerequisites"
::install_prerequisites.bat
::if ERRORLEVEL 1 GOTO errorHandling

CALL :delay 5
echo.
echo ###############^|^| Run Model Optimizer ^|^|###############
echo.
::set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
echo python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
python "%downloader_dir%\converter.py" --mo "%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer\mo.py" --name "%model_name%" -d "%models_path%" -o "%irs_path%" --precisions "%TARGET_PRECISION%"
if ERRORLEVEL 1 GOTO errorHandling


echo.
echo #######################################################################################
echo.
echo -----------------                        END                           ----------------
echo.
echo #######################################################################################
echo.

CALL :delay 5
cd /d "%ROOT_DIR%"

goto :eof

:errorHandling
echo Error
cd /d "%ROOT_DIR%"

:delay
timeout %~1 2> nul
EXIT /B 0