@echo off
REM BMI Chat Docker Stop Script for Windows
REM This script stops the BMI Chat application Docker containers

echo ========================================
echo Stopping BMI Chat Docker Environment
echo ========================================

REM Change to the docker directory
cd /d "%~dp0"

REM Stop and remove containers
echo Stopping services...
docker-compose down

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo BMI Chat stopped successfully!
    echo ========================================
    echo.
    echo To remove all data (CAUTION - this will delete your database):
    echo docker-compose down -v
    echo.
    echo To remove images as well:
    echo docker-compose down --rmi all
    echo.
) else (
    echo ERROR: Failed to stop services
)

pause
