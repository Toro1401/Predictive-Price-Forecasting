@echo off
REM ─── Kronos Daily Forecast ─── Run via Windows Task Scheduler ───
REM This script runs the multi-ticker forecast and emails results.

cd /d "%~dp0"
echo [%date% %time%] Starting Kronos forecast... >> forecast_log.txt

.venv\Scripts\python.exe forecast_all.py --all --email >> forecast_log.txt 2>&1

echo [%date% %time%] Forecast complete. >> forecast_log.txt
echo. >> forecast_log.txt
