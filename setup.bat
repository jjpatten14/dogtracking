@echo off
echo ========================================
echo    Dog Tracking System Setup
echo ========================================
echo.

echo [1/6] Checking virtual environment...
if defined VIRTUAL_ENV (
    echo Virtual environment already active: %VIRTUAL_ENV%
    echo Skipping venv creation...
) else (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        echo Make sure Python is installed and accessible
        pause
        exit /b 1
    )
    echo.
    echo [2/6] Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo [3/6] Upgrading pip...
python -m pip install --upgrade pip --verbose

echo.
echo [4/6] Installing core dependencies...
echo Installing Flask...
pip install flask --verbose
echo Installing OpenCV...
pip install opencv-python --verbose
echo Installing NumPy...
pip install numpy --verbose
echo Installing Pillow...
pip install pillow --verbose
echo Installing Requests...
pip install requests --verbose

echo.
echo [5/6] Installing additional web dependencies...
echo Installing Flask-SocketIO...
pip install flask-socketio --verbose
echo Installing Eventlet...
pip install eventlet --verbose

echo.
echo [6/8] Installing AI/ML dependencies...
echo Installing PyTorch Wildlife (MegaDetector)...
pip install PytorchWildlife --verbose
echo Installing InsightFace (Face Recognition)...
pip install insightface --verbose
echo Installing psutil (System Monitoring)...
pip install psutil --verbose
echo Installing pynvml (GPU Monitoring)...
pip install pynvml --verbose
echo Installing wandb (MiewID Training)...
pip install wandb --verbose
echo Installing Grounding DINO (Static Reference Point Detection)...
pip install transformers --verbose

echo.
echo [7/8] Installing MiewID (Animal Re-ID)...
if not exist "models" mkdir models
cd models
if not exist "wbia-plugin-miew-id" (
    echo Cloning MiewID repository...
    git clone https://github.com/WildMeOrg/wbia-plugin-miew-id
)
cd wbia-plugin-miew-id
pip install -e . --verbose
cd ..\..

echo.
echo [8/8] Creating project structure...
mkdir templates 2>nul
mkdir static 2>nul
mkdir static\css 2>nul
mkdir static\js 2>nul
mkdir static\images 2>nul
mkdir dogs 2>nul
mkdir models 2>nul

echo.
echo ========================================
echo    Setup Complete!
echo ========================================
echo Virtual environment created and dependencies installed.
echo.
echo IMPORTANT: This system uses dynamic resolution detection.
echo Cameras will use their native resolution for optimal performance.
echo Boundary coordinates automatically adapt to any resolution.
echo.
echo To start development:
echo 1. Run: call venv\Scripts\activate.bat
echo 2. Run: python app.py
echo.
pause