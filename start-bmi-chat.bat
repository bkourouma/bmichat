@echo off
setlocal enabledelayedexpansion

echo ========================================
echo       BMI Chat System Startup
echo ========================================
echo.

:: Set colors for better visibility
color 0A

:: Change to project directory
cd /d "%~dp0"

echo [1/5] Stopping existing processes on ports 3004 and 3006...
echo.

:: Kill processes on port 3006 (Backend)
echo Checking port 3006 (Backend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3006') do (
    set pid=%%a
    if defined pid (
        echo Found process !pid! on port 3006, terminating...
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✅ Successfully stopped process !pid!
        ) else (
            echo ⚠️  Could not stop process !pid!
        )
    )
)

:: Kill processes on port 3004 (Frontend)
echo Checking port 3004 (Frontend)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3004') do (
    set pid=%%a
    if defined pid (
        echo Found process !pid! on port 3004, terminating...
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo ✅ Successfully stopped process !pid!
        ) else (
            echo ⚠️  Could not stop process !pid!
        )
    )
)

:: Also kill any uvicorn or node processes that might be hanging
echo.
echo [2/5] Cleaning up any hanging processes...
taskkill /F /IM "uvicorn.exe" >nul 2>&1
taskkill /F /IM "python.exe" /FI "WINDOWTITLE eq *uvicorn*" >nul 2>&1
taskkill /F /IM "node.exe" /FI "WINDOWTITLE eq *vite*" >nul 2>&1

echo ✅ Port cleanup completed
echo.

:: Wait a moment for ports to be released
echo [3/5] Waiting for ports to be released...
timeout /t 3 /nobreak >nul
echo ✅ Ready to start services
echo.

:: Check if required directories exist
echo [4/5] Verifying project structure...
if not exist "app\main.py" (
    echo ❌ Error: Backend files not found! Make sure you're in the correct directory.
    echo Current directory: %CD%
    pause
    exit /b 1
)

if not exist "frontend\package.json" (
    echo ❌ Error: Frontend files not found! Make sure frontend directory exists.
    pause
    exit /b 1
)

echo ✅ Project structure verified
echo.

:: Start the backend and frontend
echo [5/5] Starting BMI Chat System...
echo.

:: Create a new command window for the backend
echo 🚀 Starting Backend (FastAPI) on port 3006...
start "BMI Chat Backend" cmd /k "echo Starting BMI Chat Backend... && python -m uvicorn app.main:app --host 0.0.0.0 --port 3006 --reload"

:: Wait a bit for backend to start
echo ⏳ Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

:: Create a new command window for the frontend
echo 🚀 Starting Frontend (React) on port 3004...
start "BMI Chat Frontend" cmd /k "echo Starting BMI Chat Frontend... && cd frontend && npm run dev"

:: Wait a bit for frontend to start
echo ⏳ Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo     BMI Chat System Started! 🎉
echo ========================================
echo.
echo 📊 Backend API:  http://localhost:3006
echo 🌐 Frontend App: http://localhost:3004
echo.
echo 📝 Two new command windows have opened:
echo    - BMI Chat Backend (FastAPI server)
echo    - BMI Chat Frontend (React dev server)
echo.
echo 🔗 Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul

:: Open the frontend in default browser
start http://localhost:3004

echo.
echo ✅ BMI Chat System is now running!
echo.
echo 💡 Tips:
echo    - Keep both terminal windows open
echo    - The system will auto-reload on code changes
echo    - Close terminal windows to stop the servers
echo.
echo Press any key to close this window...
pause >nul
