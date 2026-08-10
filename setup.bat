@echo off
setlocal EnableDelayedExpansion
title Eyve 2.1 — Setup

echo.
echo  ████████╗██╗   ██╗██╗   ██╗███████╗
echo  ██╔════╝╚██╗ ██╔╝██║   ██║██╔════╝
echo  █████╗   ╚████╔╝ ██║   ██║█████╗
echo  ██╔══╝    ╚██╔╝  ╚██╗ ██╔╝██╔══╝
echo  ███████╗   ██║    ╚████╔╝ ███████╗
echo  ╚══════╝   ╚═╝     ╚═══╝  ╚══════╝
echo.
echo  Visual Inspection Platform  v2.1
echo  ─────────────────────────────────
echo.

:: ── Check Python ────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo.
    echo  Please install Python 3.10 or newer from:
    echo    https://www.python.org/downloads/
    echo.
    echo  Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%

:: ── Check version >= 3.10 ────────────────────────────────────────────────────
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if errorlevel 1 (
    echo  [ERROR] Python 3.10+ is required. Found %PYVER%.
    pause
    exit /b 1
)

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
    echo  Virtual environment already exists.
)

:: ── Activate venv ────────────────────────────────────────────────────────────
call .venv\Scripts\activate.bat

echo.
echo  [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo  [3/4] Installing dependencies...
echo  This may take several minutes (ultralytics + torch download ~1-2 GB)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo  [ERROR] Some dependencies failed to install.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo  [4/4] Verifying installation...
python -c "import customtkinter, cv2, ultralytics, PIL, yaml; print('  All dependencies OK')"
if errorlevel 1 (
    echo  [WARNING] Some packages may not have installed correctly.
)

:: ── Create run shortcut ──────────────────────────────────────────────────────
echo.
echo  Creating run.bat launcher...
(
    echo @echo off
    echo cd /d "%~dp0"
    echo call .venv\Scripts\activate.bat
    echo python -m eyve.main
) > run.bat

echo.
echo  ────────────────────────────────────────
echo  Setup complete!
echo.
echo  To start Eyve:
echo    Double-click run.bat
echo    or run: python -m eyve.main
echo.
echo  First launch will show a 30-day trial.
echo  After that, Eyve remains fully functional
echo  with a license reminder (like WinRAR).
echo  ────────────────────────────────────────
echo.
pause
