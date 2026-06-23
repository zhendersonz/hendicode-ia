@echo off
title HendiCode Rollback
echo ====================================
echo  HendiCode - Restaurando Backup
echo ====================================
echo.

echo [1/4] Restaurando modelo...
xcopy /E /I /Y "%~dp0backup_antes_merge\hendicode" "%~dp0models\hendicode"

echo [2/4] Restaurando agent/model.py...
copy /Y "%~dp0backup_antes_merge\model.py.bak" "%~dp0agent\model.py"

echo [3/4] Restaurando agent/core.py...
copy /Y "%~dp0backup_antes_merge\core.py.bak" "%~dp0agent\core.py"

echo [4/4] Restaurando app.py...
copy /Y "%~dp0backup_antes_merge\app.py.bak" "%~dp0app.py"

echo.
echo Backup restaurado com sucesso!
pause
