@echo off
title Dog Tracking System - Web Server

echo.
echo   ========================================================
echo                    DOG TRACKING SYSTEM                    
echo                      Web Server v1.0                     
echo   ========================================================
echo.

echo   - Initializing virtual environment...
call venv\Scripts\activate.bat >nul 2>&1
if %errorlevel% neq 0 (
    echo   X Virtual environment not found
    echo   - Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo   - Verifying Flask installation...
python -c "import flask; print('   - Flask', flask.__version__, 'ready')" 2>nul
if %errorlevel% neq 0 (
    echo   X Flask not installed
    echo   - Please run setup.bat first
    echo.
    pause
    exit /b 1
)

echo   - Checking TTS dependencies...
python -c "import soundfile; print('   - soundfile ready')" 2>nul
if %errorlevel% neq 0 (
    echo   - TTS may not work: pip install soundfile kokoro
) else (
    echo   - TTS dependencies appear ready
)

echo   - Starting web server...
echo.
echo   ========================================================
echo     Server: http://localhost:5000                        
echo     Status: RUNNING                                      
echo     Press Ctrl+C to stop                                
echo   ========================================================
echo.

python app.py