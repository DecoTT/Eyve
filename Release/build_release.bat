@echo off
setlocal EnableDelayedExpansion
title Eyve 2.1 Beta - Build Release Package

set VERSION=2.1_Beta
set DIST_NAME=Eyve_%VERSION%
set DIST_DIR=%~dp0%DIST_NAME%
set SOURCE_DIR=%~dp0..

echo.
echo  Building Eyve %VERSION% release package...
echo  Source : %SOURCE_DIR%
echo  Output : %DIST_DIR%
echo.

:: Remove previous build
if exist "%DIST_DIR%" (
    echo  Removing previous build...
    rmdir /s /q "%DIST_DIR%"
)
mkdir "%DIST_DIR%"

:: Copy source via PowerShell (handles spaces in path, excludes __pycache__ / .pyc)
echo  Copying eyve package...
powershell -NoProfile -Command ^
  "$src='%SOURCE_DIR%\eyve'; $dst='%DIST_DIR%\eyve';" ^
  "Get-ChildItem $src -Recurse |" ^
  "Where-Object { $_.FullName -notmatch '__pycache__' -and $_.Extension -notin '.pyc','.log' } |" ^
  "ForEach-Object {" ^
  "  $rel = $_.FullName.Substring($src.Length);" ^
  "  $target = $dst + $rel;" ^
  "  if ($_.PSIsContainer) { New-Item -ItemType Directory -Force -Path $target | Out-Null }" ^
  "  else { Copy-Item $_.FullName -Destination $target -Force }" ^
  "}"
if errorlevel 1 (
    echo  [ERROR] Failed to copy eyve package.
    pause
    exit /b 1
)

:: Copy root distribution files
echo  Copying root files...
copy "%SOURCE_DIR%\requirements.txt"        "%DIST_DIR%\" >nul
copy "%SOURCE_DIR%\LICENSE"                 "%DIST_DIR%\" >nul 2>&1
copy "%SOURCE_DIR%\THIRD_PARTY_NOTICES.md"  "%DIST_DIR%\" >nul 2>&1

:: Copy installer scripts + tester README from Release/
copy "%~dp0setup.bat"  "%DIST_DIR%\setup.bat"  >nul
copy "%~dp0run.bat"    "%DIST_DIR%\run.bat"    >nul
copy "%~dp0README.md"  "%DIST_DIR%\README.md"  >nul

:: Create empty projects folder
mkdir "%DIST_DIR%\projects"
echo. > "%DIST_DIR%\projects\.gitkeep"

:: Create ZIP with PowerShell
echo.
set ZIP_OUT=%~dp0%DIST_NAME%.zip
if exist "%ZIP_OUT%" del "%ZIP_OUT%"
echo  Creating %DIST_NAME%.zip...
powershell -NoProfile -Command "Compress-Archive -Path '%DIST_DIR%\*' -DestinationPath '%ZIP_OUT%' -Force"
if errorlevel 1 (
    echo  [WARNING] ZIP creation failed. The folder is ready at:
    echo    %DIST_DIR%
) else (
    echo  ZIP created: %ZIP_OUT%
)

:: SHA-256 for the download page (SmartScreen mitigation — README tells
:: testers how to verify the ZIP they downloaded matches this hash)
echo.
echo  SHA-256:
powershell -NoProfile -Command ^
  "(Get-FileHash '%ZIP_OUT%' -Algorithm SHA256).Hash"

echo.
echo  ----------------------------------------------------
echo   Release package ready!
echo.
echo   Folder : %DIST_DIR%
echo   ZIP    : %ZIP_OUT%
echo.
echo   Share the ZIP. Users unzip and run setup.bat
echo  ----------------------------------------------------
echo.
pause
