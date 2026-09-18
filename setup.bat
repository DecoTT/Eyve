@echo off
setlocal EnableDelayedExpansion
title Eyve 2.1 - Setup

echo.
echo  +-----------------------------------------+
echo  ^|      Eyve 2.1  -  Visual Inspection     ^|
echo  +-----------------------------------------+
echo.

cd /d "%~dp0"

:: --- Check Python (install it automatically if missing) ---------------------
call :find_python
if defined PY goto :python_ok

echo  Python was not found on this computer.
echo.
echo  Eyve needs Python 3.12 to run. It can be installed automatically
echo  (about 30 MB, takes 1-2 minutes).
echo.
set /p INSTALLPY=  Install Python now? [Y/n]:
if /i "%INSTALLPY%"=="n" (
    echo.
    echo  Cancelled. You can install Python manually from:
    echo    https://www.python.org/downloads/
    echo  Remember to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: winget ships with Windows 10 21H2+ and Windows 11
where winget >nul 2>&1
if errorlevel 1 goto :no_winget

echo.
echo  Installing Python 3.12 via winget...
winget install -e --id Python.Python.3.12 --scope machine --accept-source-agreements --accept-package-agreements
if errorlevel 1 goto :no_winget

:: winget updates the PATH of NEW processes only - find the fresh install
call :find_python
if defined PY goto :python_ok
echo.
echo  Python was installed but is not visible in this window yet.
echo  Close this window and run setup.bat again to continue.
echo.
pause
exit /b 0

:no_winget
echo.
echo  [ERROR] Could not install Python automatically.
echo.
echo  Please install it manually from:
echo    https://www.python.org/downloads/
echo.
echo  IMPORTANT: check "Add Python to PATH" during installation,
echo  then run setup.bat again.
echo.
start https://www.python.org/downloads/
pause
exit /b 1

:python_ok
for /f "tokens=2" %%v in ('"%PY%" --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%

"%PY%" -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
if errorlevel 1 (
    echo  [ERROR] Python 3.10 or newer required. Found %PYVER%.
    echo  Download: https://www.python.org/downloads/
    start https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Create virtual environment
echo.
echo  [1/4] Creating virtual environment...
if not exist ".venv" (
    "%PY%" -m venv .venv
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
echo   To launch Eyve 2.1:
echo     Double-click  run.bat
echo.
echo   First launch: open or create a project.
echo   Models download on first use (~15 MB).
echo  ============================================
echo.
pause
exit /b 0


:: --- Subroutine: locate a usable Python -------------------------------------
:: Sets PY to the interpreter path, or leaves it undefined.
:: The py launcher is checked first: it finds installs that are NOT on PATH,
:: which is the most common state after a winget/Store install.
:find_python
set "PY="
:: Resolve to a real executable path (never "py -3": quoting a two-token
:: value breaks every "%PY%" call site).
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY exit /b 0
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%P"
)
if defined PY exit /b 0
for %%D in (
    "%LocalAppData%\Programs\Python\Python313\python.exe"
    "%LocalAppData%\Programs\Python\Python312\python.exe"
    "%LocalAppData%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
) do (
    if exist %%D (set "PY=%%~D" & exit /b 0)
)
exit /b 1
