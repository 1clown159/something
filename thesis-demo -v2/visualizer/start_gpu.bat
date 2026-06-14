@echo off
title Stage4 Visualizer

set ANACONDA_PYTHON=C:\ProgramData\anaconda3\python.exe
set BASE_DIR=C:\Users\32599\Desktop\thesis-demo\visualizer

echo ============================================
echo   Stage4 Visualizer - GPU Mode
echo ============================================
echo.

REM ---- Kill any process hogging port 8000 ----
echo [*] Cleaning port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING"') do (
    echo [*] Killing PID %%a on port 8000...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM ---- Kill frontend port 8080 ----
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080.*LISTENING"') do (
    echo [*] Killing PID %%a on port 8080...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo.
echo [+] Backend:  http://localhost:8000
echo [+] Frontend: http://localhost:8080
echo [+] API Docs: http://localhost:8000/docs
echo.

echo [*] Starting backend with GPU...
REM Use cmd /k to keep window open even on error, redirect stderr to a log
start "Stage4-Backend" cmd /k "cd /d "%BASE_DIR%\backend" && "%ANACONDA_PYTHON%" app.py 2>&1 || echo ERROR: Backend crashed! Check above for details. && pause"

echo [*] Waiting 4s for backend init...
timeout /t 4 /nobreak >nul

echo [*] Starting frontend...
start "Stage4-Frontend" cmd /k "cd /d "%BASE_DIR%\frontend" && "%ANACONDA_PYTHON%" -m http.server 8080"

echo.
echo ============================================
echo   All services started!
echo   Open http://localhost:8080
echo ============================================
echo.
pause
