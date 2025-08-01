@echo off
setlocal enabledelayedexpansion

echo ========================================
echo       BMI Chat System Shutdown
echo ========================================
echo.

:: Set colors for better visibility
color 0C

echo [1/3] Stopping BMI Chat System...
echo.

:: Kill processes on port 3006 (Backend)
echo 🛑 Stopping Backend (port 3006)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3006') do (
    set pid=%%a
    if defined pid (
        echo Found process !pid! on port 3006, terminating...
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✅ Successfully stopped backend process !pid!
        ) else (
            echo ⚠️  Could not stop process !pid!
        )
    )
)

:: Kill processes on port 3004 (Frontend)
echo 🛑 Stopping Frontend (port 3004)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3004') do (
    set pid=%%a
    if defined pid (
        echo Found process !pid! on port 3004, terminating...
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✅ Successfully stopped frontend process !pid!
        ) else (
            echo ⚠️  Could not stop process !pid!
        )
    )
)

echo.
echo [2/3] Cleaning up related processes...

:: Kill any uvicorn processes
echo 🧹 Stopping uvicorn processes...
taskkill /F /IM "uvicorn.exe" >nul 2>&1
if !errorlevel! equ 0 (
    echo ✅ Uvicorn processes stopped
) else (
    echo ℹ️  No uvicorn processes found
)

:: Kill any python processes running uvicorn
echo 🧹 Stopping Python/uvicorn processes...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr uvicorn') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Kill any node processes running vite
echo 🧹 Stopping Node/Vite processes...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq node.exe" /FO CSV ^| findstr vite') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: Close any command windows with BMI Chat titles
echo 🧹 Closing BMI Chat terminal windows...
taskkill /F /FI "WINDOWTITLE eq BMI Chat Backend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq BMI Chat Frontend*" >nul 2>&1

echo.
echo [3/3] Verifying shutdown...
timeout /t 2 /nobreak >nul

:: Check if ports are now free
echo 🔍 Checking port status...
netstat -aon | findstr :3006 >nul
if !errorlevel! equ 0 (
    echo ⚠️  Port 3006 still in use
) else (
    echo ✅ Port 3006 is free
)

netstat -aon | findstr :3004 >nul
if !errorlevel! equ 0 (
    echo ⚠️  Port 3004 still in use
) else (
    echo ✅ Port 3004 is free
)

echo.
echo ========================================
echo    BMI Chat System Stopped! 🛑
echo ========================================
echo.
echo ✅ All BMI Chat processes have been terminated
echo ✅ Ports 3004 and 3006 are now available
echo.
echo Press any key to close this window...
pause >nul
