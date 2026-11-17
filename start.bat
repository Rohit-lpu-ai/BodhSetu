@echo off
REM ===================================================================
REM  NiyamSetu / BodhSetu - start the whole application in one click.
REM
REM  One process, one port: the FastAPI application serves both the API
REM  (/api) and the frontend (/), so there is nothing else to start and
REM  no CORS to configure.
REM
REM  What this script checks before starting, in order, so a failure
REM  names its own cause instead of appearing as a blank page:
REM    1. the project virtualenv exists and has the dependencies
REM    2. .env exists (it holds the database password; never committed)
REM    3. PostgreSQL answers
REM    4. the schema migrations are applied
REM    5. port availability (auto-frees stale process if port is occupied)
REM    6. launches server and opens the frontend in the default browser
REM
REM  Usage:  start.bat              (port 8000, opens frontend)
REM          start.bat 8080         (another port)
REM          start.bat 8000 chat.html (specific frontend page)
REM ===================================================================

setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

set "PAGE=%~2"

set "PY=%~dp0.venv\Scripts\python.exe"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"

echo.
echo   NiyamSetu - starting up
echo   ------------------------------------------------------------
echo.

REM --- 1. virtualenv -------------------------------------------------
if not exist "%PY%" (
  echo   [X] No virtualenv at .venv\
  echo.
  echo       Create it and install the pinned dependencies:
  echo         python -m venv .venv
  echo         .venv\Scripts\python -m pip install -r requirements.txt
  echo.
  goto :fail
)

"%PY%" -c "import fastapi, uvicorn, sqlalchemy, psycopg, pypdf" >nul 2>&1
if errorlevel 1 (
  echo   [X] The virtualenv is missing dependencies.
  echo.
  echo       Install them:
  echo         .venv\Scripts\python -m pip install -r requirements.txt
  echo.
  goto :fail
)
echo   [ok] virtualenv and dependencies

REM --- 2. .env -------------------------------------------------------
if not exist ".env" (
  echo   [X] No .env file.
  echo.
  echo       Copy the example and fill in your PostgreSQL password:
  echo         copy .env.example .env
  echo.
  goto :fail
)
echo   [ok] .env present

REM --- 3. database ---------------------------------------------------
"%PY%" -c "from backend.app.db.engine import ping; import sys; sys.exit(0 if ping() else 1)" >nul 2>&1
if errorlevel 1 (
  echo   [X] PostgreSQL did not answer.
  echo.
  echo       Check that the service is running:
  echo         sc query postgresql-x64-18
  echo       and that NIYAMSETU_DB_* in .env are correct. If the database
  echo       named there does not exist yet, create it:
  echo         createdb -U postgres niyamsetu_app
  echo.
  goto :fail
)
echo   [ok] PostgreSQL reachable

REM --- 4. migrations -------------------------------------------------
"%PY%" -m backend.app.db.migrate
if errorlevel 1 (
  echo   [X] The schema migrations could not be applied. See the message above.
  goto :fail
)
echo   [ok] schema up to date

REM --- 5. port check & auto-free --------------------------------------
netstat -ano | findstr /R /C:"LISTENING" | findstr /C:":%PORT% " >nul 2>&1
if not errorlevel 1 (
  echo   [!] Port %PORT% is in use by an existing process.
  echo       Stopping previous process to free port %PORT%...
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:"LISTENING" ^| findstr /C:":%PORT% "') do (
    taskkill /F /T /PID %%a >nul 2>&1
  )
  "%PY%" -c "import time; time.sleep(1)" >nul 2>&1
  netstat -ano | findstr /R /C:"LISTENING" | findstr /C:":%PORT% " >nul 2>&1
  if not errorlevel 1 (
    echo   [X] Port %PORT% could not be freed.
    echo       Close whatever is using it, or start on another port:
    echo         start.bat 8080
    echo.
    goto :fail
  )
  echo   [ok] port %PORT% freed
) else (
  echo   [ok] port %PORT% is free
)

echo.
echo   ------------------------------------------------------------
echo     Frontend Landing     http://127.0.0.1:%PORT%/
echo     User Home / App      http://127.0.0.1:%PORT%/user_home.html
echo     Explore Policies     http://127.0.0.1:%PORT%/explore.html
echo     Chat / Ask Policy    http://127.0.0.1:%PORT%/chat.html
echo     Add a Document       http://127.0.0.1:%PORT%/add-document.html
echo     API Documentation    http://127.0.0.1:%PORT%/docs
echo   ------------------------------------------------------------
echo     Opening frontend in your browser...
echo     Press Ctrl+C in this window to stop the server.
echo.

REM Open the browser once the server is listening and answering health checks.
start "" "%PYW%" -m backend.scripts.open_browser %PORT% %PAGE%

"%PY%" -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port %PORT%

echo.
echo   NiyamSetu stopped.
endlocal
exit /b 0

:fail
echo   ------------------------------------------------------------
echo   Startup aborted. Nothing was changed.
echo.
pause
endlocal
exit /b 1
