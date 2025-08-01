@echo off
REM BMI Chat Docker Start Script for Windows
REM This script starts the BMI Chat application using Docker Compose

echo ========================================
echo Starting BMI Chat Docker Environment
echo ========================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Change to the docker directory
cd /d "%~dp0"

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found. Copying from .env.docker template...
    copy ".env.docker" ".env"
    echo.
    echo IMPORTANT: Please edit .env file and add your OpenAI API key!
    echo Press any key to continue after updating the .env file...
    pause
)

REM Build and start the services
echo Building and starting services...
docker-compose up --build -d

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo BMI Chat is starting up!
    echo ========================================
    echo.
    echo Frontend: http://localhost:3003
    echo Backend API: http://localhost:3006
    echo API Docs: http://localhost:3006/docs
    echo.
    echo With Nginx Proxy (optional):
    echo Full App: http://localhost:80
    echo.
    echo To start with nginx proxy, run:
    echo docker-compose --profile production up -d
    echo.
    echo To view logs: docker-compose logs -f
    echo To stop: docker-compose down
    echo.
) else (
    echo ERROR: Failed to start services
    pause
    exit /b 1
)

pause
