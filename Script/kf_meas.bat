@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_meas.bat  -  Task D: train a measurement-column fleet, then the
::                        same x1.15 four-way compare as Task A
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_meas.bat
::   working directory is forced to the repo root
::
:: command actually run
::   python Src/AI/MLP/train.py --scheme B --epochs 100 --data-dir Data/grid --out-dir Data/ai_mlp_meas --use-meas-inputs
::   python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare_meas --epochs 10 --replay-n 50
::
:: notes
::   does NOT touch Data/ai_mlp or Data/ai_kf/compare
::   Data/soh_k115 must already exist (Task A); do not --make-new
::   if hole-fleet-like val RMSE is still high, resume then compare only:
::     python Src/AI/MLP/train.py --scheme B --epochs 200 --data-dir Data/grid --out-dir Data/ai_mlp_meas --use-meas-inputs --resume
::     python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare_meas --epochs 10 --replay-n 50
:: =============================================================================

python Src/AI/MLP/train.py --scheme B --epochs 100 --data-dir Data/grid --out-dir Data/ai_mlp_meas --use-meas-inputs
if errorlevel 1 goto :done
python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare_meas --epochs 10 --replay-n 50
:done
pause
