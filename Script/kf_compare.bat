@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_compare.bat  -  four-way increment comparison (frozen / retrain / replay /
::                    finetune / scale) on a resistance-scaled grid
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_compare.bat
::   working directory is forced to the repo root
::
:: prerequisites
::   Data/ai_mlp_100/best.pt + scaler.json   (100-epoch true-column fleet)
::   Data/grid/   (BOL replay / old-set eval; do not regenerate into this folder)
::   Data/soh_k115/  already generated; do not --make-new
::
:: command actually run
::   python Src/AI/KF/compare.py --mlp-dir Data/ai_mlp_100 --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare --epochs 10 --replay-n 50
::
:: notes
::   writes Data/ai_kf/compare/     compare.md / compare.csv / per-mode weights
::   does NOT rewrite Data/soh_k115 or Data/ai_mlp
::   smoke (tiny 2x2, 1 epoch):
::     python Src/AI/KF/compare.py --smoke
::   see Src/AI/KF/readme.md and Plan.md
:: =============================================================================

python Src/AI/KF/compare.py --mlp-dir Data/ai_mlp_100 --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare --epochs 10 --replay-n 50
pause
