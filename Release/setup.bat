@echo off
setlocal EnableDelayedExpansion
title Eyve 2.1 Beta - Setup

echo.
echo  +-----------------------------------------+
echo  |   Eyve 2.1 Beta  -  Visual Inspection   |
echo  +-----------------------------------------+
echo.

cd /d "%~dp0"

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%

python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if errorlevel 1 (
    echo  [ERROR] Python 3.10 or newer required. Found %PYVER%.
    echo  Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment
echo.
echo  [1/4] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Virtual environment created.
) else (
    echo  Virtual environment already exists - skipping.
)
call .venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo  [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install dependencies
echo.
echo  [3/4] Installing dependencies...
echo  This may take several minutes (ultralytics + torch = ~1-2 GB).
echo  Do NOT close this window.
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Some dependencies failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

:: Verify
echo.
echo  [4/4] Verifying installation...
python -c "import customtkinter, cv2, ultralytics, PIL, yaml; print('  All dependencies OK')"
if errorlevel 1 (
    echo  [WARNING] Some packages may not be installed correctly.
    echo  Try running setup.bat again.
)

:: Create run.bat launcher
echo.
echo  Creating run.bat launcher...
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo call .venv\Scripts\activate.bat
    echo python -m eyve.main
    echo pause
) > run.bat

echo.
echo  ============================================
echo   Setup complete!
echo.
echo   To launch Eyve 2.1 Beta:
echo     Double-click  run.bat
echo.
echo   First launch: open or create a project.
echo   Models download on first use (~15 MB).
echo  ============================================
echo.
pause
