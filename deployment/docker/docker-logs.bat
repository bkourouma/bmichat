@echo off
REM BMI Chat Docker Logs Script for Windows
REM This script shows logs from the BMI Chat Docker containers

echo ========================================
echo BMI Chat Docker Logs
echo ========================================

REM Change to the docker directory
cd /d "%~dp0"

REM Check if services are running
docker-compose ps

echo.
echo Choose an option:
echo 1. View all logs (follow)
echo 2. View backend logs only
echo 3. View frontend logs only
echo 4. View nginx logs only
echo 5. View all logs (last 100 lines)
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Showing all logs (Ctrl+C to exit)...
    docker-compose logs -f
) else if "%choice%"=="2" (
    echo Showing backend logs (Ctrl+C to exit)...
    docker-compose logs -f backend
) else if "%choice%"=="3" (
    echo Showing frontend logs (Ctrl+C to exit)...
    docker-compose logs -f frontend
) else if "%choice%"=="4" (
    echo Showing nginx logs (Ctrl+C to exit)...
    docker-compose logs -f nginx
) else if "%choice%"=="5" (
    echo Showing last 100 lines of all logs...
    docker-compose logs --tail=100
) else (
    echo Invalid choice. Showing all logs...
    docker-compose logs -f
)

pause
