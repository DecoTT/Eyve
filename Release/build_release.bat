@echo off
title Eyve 2.1 Beta - Build Release Package
:: Lanzador delgado: toda la logica esta en build_release.ps1.
:: Motivo: los bloques `powershell -Command ^` multilinea dentro de un .bat
:: se rompen con carets + comillas + rutas con espacios (corrompio un build).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1"
set RC=%ERRORLEVEL%
pause
exit /b %RC%
