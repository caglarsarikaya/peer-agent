@echo off
echo Stopping Peer Agent System containers...
echo.

REM Stop and remove containers
docker-compose down

echo.
echo Containers stopped and removed.
echo.

REM Ask if user wants to remove volumes (data)
set /p cleanup="Remove database data volumes? (y/N): "
if /i "%cleanup%"=="y" (
    echo Removing data volumes...
    docker-compose down -v
    echo Data volumes removed.
) else (
    echo Database data preserved.
)

echo.
echo Cleanup complete. Press any key to exit.
pause 