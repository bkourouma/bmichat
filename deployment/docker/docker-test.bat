@echo off
REM BMI Chat Docker Test Script for Windows
REM This script validates the Docker setup and tests the build process

echo ========================================
echo BMI Chat Docker Setup Test
echo ========================================

REM Change to the docker directory
cd /d "%~dp0"

REM Check if Docker is installed
echo [1/6] Checking Docker installation...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed. Please install Docker Desktop first.
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)
echo ✓ Docker is installed

REM Check if Docker is running
echo [2/6] Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    echo After starting Docker Desktop, wait for it to fully initialize and try again.
    pause
    exit /b 1
)
echo ✓ Docker is running

REM Check if .env file exists and has API key
echo [3/6] Checking environment configuration...
if not exist ".env" (
    echo ERROR: .env file not found. Please copy .env.docker to .env and configure it.
    pause
    exit /b 1
)

findstr /C:"sk-proj-your-openai-api-key-here" ".env" >nul
if %errorlevel% equ 0 (
    echo WARNING: Please update your OpenAI API key in the .env file
    echo Current key appears to be the default placeholder.
    echo.
    set /p continue="Continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Please update .env file and run this script again.
        pause
        exit /b 1
    )
)
echo ✓ Environment file exists

REM Check if required source files exist
echo [4/6] Checking source files...
if not exist "..\..\app\main.py" (
    echo ERROR: Backend source files not found. Please ensure you're in the correct directory.
    pause
    exit /b 1
)
if not exist "..\..\frontend\package.json" (
    echo ERROR: Frontend source files not found. Please ensure you're in the correct directory.
    pause
    exit /b 1
)
echo ✓ Source files found

REM Test Docker Compose configuration
echo [5/6] Validating Docker Compose configuration...
docker-compose config >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker Compose configuration is invalid.
    echo Running docker-compose config to show errors:
    docker-compose config
    pause
    exit /b 1
)
echo ✓ Docker Compose configuration is valid

REM Test build process (without starting services)
echo [6/6] Testing Docker build process...
echo This may take several minutes on first run...
docker-compose build --no-cache
if %errorlevel% neq 0 (
    echo ERROR: Docker build failed. Please check the error messages above.
    pause
    exit /b 1
)
echo ✓ Docker build successful

echo.
echo ========================================
echo All tests passed! ✓
echo ========================================
echo.
echo Your Docker setup is ready. You can now:
echo.
echo 1. Start the application:
echo    docker-start.bat
echo.
echo 2. Or manually with:
echo    docker-compose up -d
echo.
echo 3. View logs:
echo    docker-logs.bat
echo.
echo 4. Stop the application:
echo    docker-stop.bat
echo.
echo Access points after starting:
echo - Frontend: http://localhost:3003
echo - Backend API: http://localhost:3006
echo - API Docs: http://localhost:3006/docs
echo.

pause
