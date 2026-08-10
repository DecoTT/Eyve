@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m eyve.main
pause
