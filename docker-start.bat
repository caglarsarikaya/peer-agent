@echo off
echo Starting Peer Agent System with Docker...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker .env file exists
if not exist .env.docker (
    echo WARNING: .env.docker file not found. 
    echo Please copy env.docker.example to .env.docker and add your API keys.
    echo.
    echo Example:
    echo   copy env.docker.example .env.docker
    echo   # Then edit .env.docker and add your OPENAI_API_KEY
    echo.
    pause
    exit /b 1
)

echo Building and starting containers...
docker-compose up --build

echo.
echo Containers stopped. Press any key to exit.
pause 