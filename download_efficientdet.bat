@echo off
echo ========================================
echo    EfficientDet-D0 Model Download
echo ========================================
echo.

echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    echo Make sure you're running from the yard directory
    pause
    exit /b 1
)

echo.
echo [2/3] Setting models directory...
set TORCH_HOME=%CD%\models
echo Models will be downloaded to: %TORCH_HOME%

echo.
echo [3/3] Downloading EfficientDet-D0 model...
echo This may take a few minutes depending on your internet connection...
python -c "import os; os.environ['TORCH_HOME'] = r'%CD%\models'; import effdet; model = effdet.create_model('efficientdet_d0', pretrained=True); print('✅ EfficientDet-D0 model downloaded successfully!')"

if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to download EfficientDet-D0 model
    echo Check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ========================================
echo    Download Complete!
echo ========================================
echo EfficientDet-D0 model is now ready for reference point detection.
echo Model location: %CD%\models
echo.
pause