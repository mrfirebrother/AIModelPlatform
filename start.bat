@echo off
echo Starting AI Model Platform...

echo.
echo [1/2] Starting Backend on port 8000...
set POSTGRES_HOST=localhost
set POSTGRES_DB=ai_platform
set POSTGRES_USER=ai_platform
set POSTGRES_PASSWORD=change-me
set REDIS_URL=redis://localhost:6379/0
set PLATFORM_API_KEY=change-me
cd /d F:\Work\EStructure\AIModelPlatform
start "Backend" cmd /c "python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

echo.
echo [2/2] Starting Frontend on port 3000...
cd /d F:\Work\EStructure\AIModelPlatform\frontend
start "Frontend" cmd /c "npx vite --host 127.0.0.1 --port 3000"

echo.
echo Both services started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause