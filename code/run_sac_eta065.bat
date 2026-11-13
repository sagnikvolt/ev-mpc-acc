@echo off
chcp 65001 >nul
title SAC eta 0.65 re-run  -  do not close until COMPLETE
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
echo === Step 1 of 3: SAC training, 3 seeds in parallel (can be closed and re-run; it resumes) ===
venv310\Scripts\python python\rerun_sac_eta065.py
echo.
echo === Step 2 of 3: clean single-core solve timing (laptop must be idle) ===
venv310\Scripts\python python\timing_solve.py
echo.
echo === Step 3 of 3: summary ===
venv310\Scripts\python python\summary_sac_eta065.py > results\eta065_rerun\summary_sac.txt 2>&1
type results\eta065_rerun\summary_sac.txt
echo.
echo === ALL DONE. You can close this window. ===
pause
