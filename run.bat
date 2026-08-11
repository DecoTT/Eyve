@echo off
rem %~dp0 = carpeta donde vive ESTE run.bat — nunca una ruta fija.
rem (El bug anterior: setup.bat genero run.bat con la ruta expandida a 2.1,
rem  asi que el run.bat copiado a 2.1-dev lanzaba el respaldo viejo.)
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m eyve.main
pause
