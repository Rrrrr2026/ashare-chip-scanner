@echo off
REM Portable launcher (any Windows PC). Requires: pip install -r requirements.txt
cd /d "%~dp0"
set "PY=py"
where py >nul 2>nul || set "PY=python"
if not exist "techscan_data\metrics.parquet" (
  echo No cached data found. Building (needs network, takes a while)...
  %PY% techscan_build.py
)
echo Starting dashboard at http://localhost:8501  - close this window to stop
%PY% -m streamlit run techscan_app.py
pause
