:: 1. Model Path must be set manually
:: 2. Output Exit Layer must be set manually

@echo off
setlocal enabledelayedexpansion

title modelSlice

echo.
echo #######################################################################################
echo.
echo ----------           Batch Script to perform Model Slicing                   ----------
echo.
echo #######################################################################################
echo.

:: Setting up default values for TARGET, BUILD_FOLDER and ROOT_DIR
set TARGET=CPU
set BUILD_FOLDER=%USERPROFILE%\Documents\openvino_models\ir\public
set ROOT_DIR=C:\Program Files (x86)\Intel\openvino_2021.3.394

:input_arguments_loop
if not "%1"=="" (
    if "%1"=="--d" (
        set TARGET=%2
        shift
    )
    if "%1"=="--model-name" (
        set MODEL_NAME=%2
        shift
    )
    if "%1"=="--precision" (
        set TARGET_PRECISION=%2
        shift
    )
    shift
    goto :input_arguments_loop
)


:: Activating the OpenVINO Environment
if exist "%ROOT_DIR%\bin\setupvars.bat" (
    call "%ROOT_DIR%\bin\setupvars.bat"
) else (
    echo setupvars.bat is not found, INTEL_OPENVINO_DIR can't be set
    goto error
)

set input_shape=[1,224,224,3]


:: Displaying Relevant Information
echo.
echo ##################################################
echo.
echo INTEL_OPENVINO_DIR is set to %INTEL_OPENVINO_DIR%
echo.
echo Target = %TARGET%
echo.
echo Model Name = %MODEL_NAME%
echo.
echo Precision = %TARGET_PRECISION%
echo.
echo Input Shape = %input_shape%
echo.
echo ##################################################
echo.

:: Setting variable to point to the model
set model_path=%USERPROFILE%\Documents\openvino_models\models\public\%model_name%\resnet_v1-50.pb

:: Setting up variable that defines the destination folder (Holds the .xml and .bin files)
set ir_dir=%BUILD_FOLDER%\%model_name%\%target_precision%

:: Setting up variable to hold location of the .mo.py script
set optimizer_dir=%INTEL_OPENVINO_DIR%\deployment_tools\model_optimizer

:: Setting up variable to hold the layer at which to exit
set output_exit=resnet_model/Squeeze

:: Building IR Representation of Sliced Model
echo cd %optimizer_dir%
echo python .\mo.py --input_model %model_path% -o %ir_dir% --input_shape=%input_shape% --output %output_exit%
cd %optimizer_dir%
python .\mo.py --input_model %model_path% -o %ir_dir% --input_shape=%input_shape% --output %output_exit%


echo.
echo #######################################################################################
echo.
echo -----------------                        END                           ----------------
echo.
echo #######################################################################################
echo.