@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_q90.bat  -  Task G: 1RC q=0.90 grid + four-way increment compare
:: =============================================================================
::
:: does not regenerate Data/grid
::   python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 5 --n-temp 5 --out-dir Data/soh_q90 --soh 0.90
::   python Src/AI/KF/compare.py --task aging --mlp-dir Data/ai_mlp --new-dir Data/soh_q90 --old-dir Data/grid --out-dir Data/ai_kf/compare_q90 --epochs 10 --replay-n 50
:: =============================================================================

python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 5 --n-temp 5 --out-dir Data/soh_q90 --soh 0.90
if errorlevel 1 goto :done
python Src/AI/KF/compare.py --task aging --mlp-dir Data/ai_mlp --new-dir Data/soh_q90 --old-dir Data/grid --out-dir Data/ai_kf/compare_q90 --epochs 10 --replay-n 50
:done
pause
