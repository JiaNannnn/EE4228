@echo off
echo Facial Recognition - Gallery Preprocessing Tool
echo =============================================
echo.

REM Create necessary directories if they don't exist
if not exist models mkdir models
if not exist data mkdir data
if not exist logs mkdir logs
if not exist data\users mkdir data\users

echo Checking for required packages...
python -c "import tqdm" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo Installing tqdm package...
    pip install tqdm
)

echo.
echo Starting gallery preprocessing with 128x128 resolution...
echo.
echo NOTE: This will reprocess all gallery images with the updated resolution.
echo.

REM Run the simple gallery processor with overwrite flag
python simple_gallery_processor.py --overwrite

echo.
echo Processing complete!
echo.
echo The images have been processed with 128x128 resolution and registered in the system.
echo You should now retrain the model using the Streamlit application.
echo.

pause 