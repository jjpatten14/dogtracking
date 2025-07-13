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
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --verbose

echo.
echo [4/5] Installing dependencies from requirements.txt...
if exist "requirements.txt" (
    echo Installing from requirements.txt for version consistency...
    pip install -r requirements.txt --verbose
    if %errorlevel% neq 0 (
        echo WARNING: requirements.txt installation failed, falling back to individual packages...
        goto :fallback_install
    )
    goto :skip_fallback
) else (
    echo requirements.txt not found, installing individual packages...
    goto :fallback_install
)

:fallback_install
echo [5/5] Installing additional core dependencies individually...
echo Installing OpenCV...
pip install opencv-python --verbose
echo Installing NumPy...
pip install numpy --verbose
echo Installing Pillow...
pip install pillow --verbose
echo Installing Requests...
pip install requests --verbose
echo Installing Eventlet...
pip install eventlet --verbose

:skip_fallback

echo.
echo [6/9] Installing PyTorch with CUDA support...
echo Installing PyTorch for GPU acceleration...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --verbose
if %errorlevel% neq 0 (
    echo WARNING: CUDA PyTorch installation failed, installing CPU version...
    pip install torch torchvision torchaudio --verbose
)

echo.
echo [7/9] Installing TTS dependencies...
echo Installing Kokoro TTS...
pip install kokoro --verbose
echo Installing SoundFile (audio I/O)...
pip install soundfile --verbose
echo Installing PyGame (audio playback)...
pip install pygame --verbose

echo.
echo [8/9] Installing AI/ML dependencies...
echo Installing PyTorch Wildlife (MegaDetector)...
pip install PytorchWildlife --verbose
echo Installing standalone MegaDetector v5...
pip install megadetector --verbose
echo Installing InsightFace (Face Recognition)...
pip install insightface --verbose
echo Installing psutil (System Monitoring)...
pip install psutil --verbose
echo Installing pynvml (GPU Monitoring)...
pip install pynvml --verbose
echo Installing CuPy with CUDA 12.x support...
pip uninstall cupy -y 2>nul
pip install cupy-cuda12x --verbose
rem Removed nvidia-ml-py3 due to conflicts with pynvml
echo Installing PyCUDA (GPU computing)...
pip install pycuda --verbose
echo Installing EfficientDet (Advanced detection)...
pip install effdet --verbose
echo Installing additional dependencies...
pip install dnspython --verbose
pip install antlr4-python3-runtime --verbose
pip install bidict --verbose
pip install greenlet --verbose
pip install simple-websocket --verbose
pip install siphash24 --verbose
pip install imageio-ffmpeg --verbose
pip install networkx --verbose
echo Installing wandb (MiewID Training)...
pip install wandb --verbose
echo Configuring wandb for offline mode...
python -c "import wandb; wandb.login(anonymous='must'); wandb.init(mode='offline')" 2>nul || echo Wandb offline config attempted...
wandb offline 2>nul || echo Wandb offline command attempted...
echo Installing Grounding DINO (Static Reference Point Detection)...
pip install transformers --verbose

echo.
echo [9/9] Installing MiewID (Animal Re-ID)...
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
echo [10/11] Creating project structure...
mkdir templates 2>nul
mkdir static 2>nul
mkdir static\css 2>nul
mkdir static\js 2>nul
mkdir static\images 2>nul
mkdir dogs 2>nul
mkdir dogs\annotations 2>nul
mkdir dogs\preprocessed 2>nul
mkdir models 2>nul
mkdir snapshots 2>nul
echo. > snapshots\.gitkeep

echo.
echo [11/11] Installing web framework...
echo Installing Flask (required for web server)...
pip install flask --verbose
echo Installing Flask-SocketIO (required for real-time features)...
pip install flask-socketio --verbose

echo.
echo ========================================
echo    Validating Installation...
echo ========================================
echo Testing critical system components...
python -c "import torch; print('✓ PyTorch:', torch.__version__); print('✓ CUDA Available:', torch.cuda.is_available())"
python -c "import pynvml; print('✓ pynvml imported successfully')" 2>nul || echo ✗ pynvml import failed
python -c "import cupy; print('✓ CuPy installed:', cupy.__version__)" 2>nul || echo ⚠ CuPy issue
python -c "import soundfile, pygame; print('✓ SoundFile and PyGame installed')" 2>nul || echo ⚠ Audio packages issue
python -c "from kokoro import KPipeline; print('✓ Kokoro TTS installed')" 2>nul || echo ✗ Kokoro TTS not found
python -c "import wandb; print('✓ wandb installed')" 2>nul || echo ⚠ wandb issue
python -c "import effdet, pycuda; print('✓ EfficientDet and PyCUDA installed')" 2>nul || echo ⚠ ML packages issue
python -c "from megadetector.detection.run_detector_batch import load_detector; print('✓ MegaDetector v5 installed')" 2>nul || echo ✗ MegaDetector import failed
echo ✓ All critical dependencies tested

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
echo Features installed:
echo - Real-time dog detection with MegaDetector
echo - Boundary violation detection and alerts
echo - Text-to-Speech with Kokoro (offline)
echo - GPU acceleration support
echo - Automatic snapshot capture
echo.
echo TROUBLESHOOTING:
echo - If you see "wandb interactive prompts": Run 'wandb offline'
echo - If you see "pynvml import errors": Check GPU drivers are installed
echo - If you see Unicode errors: Use Command Prompt (not PowerShell)
echo - All dependencies have been configured to avoid known conflicts
echo.
echo To start the system:
echo 1. Run: call venv\Scripts\activate.bat
echo 2. Run: web.bat  (recommended)
echo    OR: python app.py
echo 3. Open: http://localhost:5000
echo.
echo For production use:
echo - Disable Flask debug mode
echo - Use a production WSGI server
echo - Configure proper logging
echo.
pause