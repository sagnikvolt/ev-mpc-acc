@echo off
chcp 65001 >nul
title Safe SAC (CBF filter) eta 0.65  -  do not close until COMPLETE
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
echo === Step 1 of 2: safe-SAC training, 3 seeds in parallel (can be closed and re-run; it resumes) ===
venv310\Scripts\python python\safe_rl_train.py
echo.
echo === Step 2 of 2: summary ===
venv310\Scripts\python python\summary_safe_rl.py > results\safe_rl\summary.txt 2>&1
type results\safe_rl\summary.txt
echo.
echo === ALL DONE. You can close this window. ===
pause
