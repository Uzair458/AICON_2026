@echo off
setlocal
cd /d "%~dp0"
title Fraud Triage
set PYTHONUTF8=1

where py >nul 2>nul
if %errorlevel%==0 ( set "PY=py" ) else ( set "PY=python" )
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install it from https://www.python.org/downloads/ and tick "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a private Python environment ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo Could not create the environment. & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

if not exist ".venv\.deps_ok" (
  echo Installing packages - first run only, a few minutes ...
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Package install failed. If you use a very new Python version, install Python 3.12 or 3.13
    echo and delete the .venv folder, then run this file again. See README.md, Troubleshooting.
    pause
    exit /b 1
  )
  echo ok> ".venv\.deps_ok"
)

rem Streamlit asks for an email on first launch unless this file exists.
if not exist "%USERPROFILE%\.streamlit\credentials.toml" (
  mkdir "%USERPROFILE%\.streamlit" >nul 2>nul
  > "%USERPROFILE%\.streamlit\credentials.toml" echo [general]
  >> "%USERPROFILE%\.streamlit\credentials.toml" echo email = ""
)

echo.
echo Preparing data - the first run takes about a minute ...
python app.py --prepare
if errorlevel 1 ( echo. & pause & exit /b 1 )

echo.
echo Starting the app. A browser tab opens at http://localhost:8501  (Ctrl+C here to stop).
python -m streamlit run app.py
pause
