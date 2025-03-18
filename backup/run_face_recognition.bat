@echo off
echo ===================================
echo Real-Time Face Recognition System
echo ===================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python not found. Please install Python 3.8 or higher.
    goto :EOF
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dependencies are installed
if not exist venv\Lib\site-packages\cv2 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo Error installing dependencies. Please check requirements.txt.
        goto :deactivate
    )
    echo Dependencies installed successfully.
)

REM Check if shape predictor model exists
if not exist shape_predictor_68_face_landmarks.dat (
    echo Downloading required models...
    python download_models.py
    if %ERRORLEVEL% neq 0 (
        echo Error downloading models. 
        echo Please manually download dlib's shape_predictor_68_face_landmarks.dat model.
        goto :deactivate
    )
)

REM Display menu
:menu
cls
echo ===================================
echo Real-Time Face Recognition System
echo ===================================
echo.
echo Choose an option:
echo 1. Run real-time face recognition
echo 2. Train recognition model
echo 3. Run benchmark tests
echo 4. Exit
echo.
set /p option="Enter option (1-4): "

if "%option%"=="1" goto :run_recognition
if "%option%"=="2" goto :train_model
if "%option%"=="3" goto :run_tests
if "%option%"=="4" goto :deactivate

echo Invalid option. Please try again.
timeout /t 2 >nul
goto :menu

:run_recognition
cls
echo Running real-time face recognition...
echo Press 'q' to quit, 's' to save a face, 'd' to delete last saved face
echo.

REM Check if model exists
if not exist trained_hybrid_model.pkl (
    echo Warning: No trained model found.
    echo You will need to train the system by adding faces during runtime.
    timeout /t 3 >nul
)

python realtime_face_recognition.py
echo.
echo Recognition session ended.
timeout /t 2 >nul
goto :menu

:train_model
cls
echo Training face recognition model...
echo.

REM Ask for dataset path
set /p dataset_path="Enter dataset path (or press Enter for default AT&T dataset): "

if "%dataset_path%"=="" (
    echo Using default AT&T dataset...
    python train_hybrid_model.py --visualize
) else (
    echo Using custom dataset at %dataset_path%...
    python train_hybrid_model.py --dataset "%dataset_path%" --visualize
)

if %ERRORLEVEL% neq 0 (
    echo Error during model training.
) else (
    echo Model training completed successfully.
)

timeout /t 3 >nul
goto :menu

:run_tests
cls
echo Running benchmark tests...
echo.

echo Choose test type:
echo 1. Performance benchmark
echo 2. Accuracy test
echo 3. Stress test
echo 4. All tests
echo 5. Back to main menu
echo.
set /p test_option="Enter option (1-5): "

if "%test_option%"=="1" (
    echo Running performance benchmark...
    python test_realtime_recognition.py --test-type benchmark --duration 20
) else if "%test_option%"=="2" (
    set /p test_data="Enter test dataset path: "
    echo Running accuracy test...
    python test_realtime_recognition.py --test-type accuracy --test-data "%test_data%"
) else if "%test_option%"=="3" (
    echo Running stress test...
    python test_realtime_recognition.py --test-type stress
) else if "%test_option%"=="4" (
    set /p test_data="Enter test dataset path (optional, press Enter to skip accuracy test): "
    echo Running all tests...
    if "%test_data%"=="" (
        python test_realtime_recognition.py --test-type benchmark --duration 20
        python test_realtime_recognition.py --test-type stress
    ) else (
        python test_realtime_recognition.py --test-type benchmark --duration 20
        python test_realtime_recognition.py --test-type accuracy --test-data "%test_data%"
        python test_realtime_recognition.py --test-type stress
    )
) else if "%test_option%"=="5" (
    goto :menu
) else (
    echo Invalid option.
    timeout /t 2 >nul
    goto :run_tests
)

echo.
echo Tests completed. Results saved in test_results directory.
timeout /t 3 >nul
goto :menu

:deactivate
REM Deactivate virtual environment
call venv\Scripts\deactivate.bat
echo Exiting. Goodbye!
echo.
exit /b 0 