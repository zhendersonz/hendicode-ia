@echo off
title HendiCode Engine - Soberano
set PATH=%~dp0venv\Scripts;%PATH%
echo [HendiCode] Inicializando sistema independente...
echo [HendiCode] Local: %~dp0
echo.
python app.py
pause
