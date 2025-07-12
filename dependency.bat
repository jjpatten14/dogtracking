@echo off
title Dog Tracking System - Dependency Installation
color 0A

echo.
echo   ╔══════════════════════════════════════════════════════════════╗
echo   ║                    DOG TRACKING SYSTEM                       ║
echo   ║                  Dependency Installation                     ║
echo   ║                      Jetson Orin Nano                       ║
echo   ╚══════════════════════════════════════════════════════════════╝
echo.

echo [INFO] Starting dependency installation for Dog Tracking System...
echo [INFO] Target Hardware: Jetson Orin Nano
echo [INFO] Virtual Environment: venv
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo [ERROR] Please run setup.bat first to create the virtual environment.
    echo.
    pause
    exit /b 1
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment activated successfully!
echo.

echo [INFO] Installing core dependencies in verbose mode...
echo.

REM Core ML/AI Libraries
echo [INSTALL] PyTorch ecosystem for Jetson...
pip install --verbose torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch!
    pause
    exit /b 1
)

echo.
echo [INSTALL] MegaDetector via PyTorch Wildlife...
pip install --verbose PytorchWildlife
if errorlevel 1 (
    echo [ERROR] Failed to install PytorchWildlife!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Computer Vision libraries...
pip install --verbose opencv-python opencv-contrib-python
if errorlevel 1 (
    echo [ERROR] Failed to install OpenCV!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Web framework and APIs...
pip install --verbose Flask Flask-SocketIO Flask-CORS
if errorlevel 1 (
    echo [ERROR] Failed to install Flask!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Numerical computing libraries...
pip install --verbose numpy pandas scipy
if errorlevel 1 (
    echo [ERROR] Failed to install numerical libraries!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Image processing and utilities...
pip install --verbose Pillow scikit-image matplotlib
if errorlevel 1 (
    echo [ERROR] Failed to install image processing libraries!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Async and networking libraries...
pip install --verbose eventlet requests urllib3 websockets
if errorlevel 1 (
    echo [ERROR] Failed to install networking libraries!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Machine Learning utilities...
pip install --verbose scikit-learn joblib tqdm
if errorlevel 1 (
    echo [ERROR] Failed to install ML utilities!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Face recognition and ArcFace dependencies...
pip install --verbose face-recognition insightface onnxruntime
if errorlevel 1 (
    echo [WARNING] Some face recognition libraries failed - may need manual installation
)

echo.
echo [INSTALL] Additional utilities...
pip install --verbose python-dotenv configparser pathlib2 jsonschema
if errorlevel 1 (
    echo [ERROR] Failed to install utility libraries!
    pause
    exit /b 1
)

echo.
echo [INSTALL] Jetson-specific optimizations...
pip install --verbose jetson-stats
if errorlevel 1 (
    echo [WARNING] jetson-stats failed - this is normal on non-Jetson systems
)

echo.
echo [INFO] Verifying installation...
echo.

REM Verify critical imports
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import PIL; print(f'Pillow: {PIL.__version__}')"

echo.
echo [INFO] Attempting to verify PyTorchWildlife...
python -c "
try:
    from PytorchWildlife.models import detection as pw_detection
    print('PyTorchWildlife: Available ✓')
except ImportError as e:
    print(f'PyTorchWildlife: Import failed - {e}')
"

echo.
echo   ╔══════════════════════════════════════════════════════════════╗
echo   ║                    INSTALLATION COMPLETE                     ║
echo   ╚══════════════════════════════════════════════════════════════╝
echo.
echo [SUCCESS] All dependencies installed successfully!
echo.
echo [NEXT STEPS]
echo 1. Download required models (see models.md)
echo 2. Configure IP cameras in settings
echo 3. Run web.bat to start the system
echo.
echo [JETSON OPTIMIZATION NOTES]
echo - Ensure CUDA 11.8+ is installed
echo - Consider TensorRT model optimization
echo - Monitor GPU memory usage with jetson-stats
echo - Use jtop command for system monitoring
echo.
pause