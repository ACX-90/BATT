@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_meas.bat  -  Task D: raised-noise measurement-column fleet, then
::                        the same x1.15 four-way compare as Task A
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_meas.bat
::   working directory is forced to the repo root
::
:: noise  (written to Data/grid_noisy and Data/soh_k115_noisy; does NOT
::         rewrite Data/grid or Data/soh_k115)
::   voltage 7 mV   -- aligned to the ~7 mV floor of a 100-epoch true-column fleet
::   current 100 mA
::   temp    0.5 C
::   SOC     0.5 pp
::
:: command actually run
::   python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 10 --n-temp 10 --out-dir Data/grid_noisy --noise-voltage 0.007 --noise-current 0.1 --noise-temp 0.5 --noise-soc 0.005
::   python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 5 --n-temp 5 --out-dir Data/soh_k115_noisy --r0-scale 1.15 --r1-scale 1.15 --noise-voltage 0.007 --noise-current 0.1 --noise-temp 0.5 --noise-soc 0.005
::   python Src/AI/MLP/train.py --scheme B --epochs 100 --data-dir Data/grid_noisy --out-dir Data/ai_mlp_meas --use-meas-inputs --fresh
::   python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115_noisy --old-dir Data/grid_noisy --out-dir Data/ai_kf/compare_meas --epochs 10 --replay-n 50
::
:: notes
::   does NOT touch Data/ai_mlp, Data/grid, Data/soh_k115, or Data/ai_kf/compare
::   100 voltage epochs on purpose: no-noise 100-epoch floor is ~7 mV; do not
::   chase the 4 mV of a 500+ epoch true-column fleet
::   if grids already exist, skip generation and train / compare only
:: =============================================================================

python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 10 --n-temp 10 --out-dir Data/grid_noisy --noise-voltage 0.007 --noise-current 0.1 --noise-temp 0.5 --noise-soc 0.005
if errorlevel 1 goto :done
python Src/Sim/nmc100ah_ecm_gen_grid.py --n-soc 5 --n-temp 5 --out-dir Data/soh_k115_noisy --r0-scale 1.15 --r1-scale 1.15 --noise-voltage 0.007 --noise-current 0.1 --noise-temp 0.5 --noise-soc 0.005
if errorlevel 1 goto :done
python Src/AI/MLP/train.py --scheme B --epochs 100 --data-dir Data/grid_noisy --out-dir Data/ai_mlp_meas --use-meas-inputs --fresh
if errorlevel 1 goto :done
python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115_noisy --old-dir Data/grid_noisy --out-dir Data/ai_kf/compare_meas --epochs 10 --replay-n 50
:done
pause
