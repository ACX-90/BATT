@echo off
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
::   Data/ai_mlp/best.pt + scaler.json
::   Data/grid/   (BOL replay / old-set eval; do not regenerate into this folder)
::
:: command actually run
::   python Src/AI/KF/compare.py --make-new --r0-scale 1.15 --r1-scale 1.15 --epochs 10
::
:: notes
::   writes Data/soh_k115/          scaled new-year grid
::   writes Data/ai_kf/compare/     compare.md / compare.csv / per-mode weights
::   smoke (tiny 2x2, 1 epoch):
::     python Src/AI/KF/compare.py --smoke
::   see Src/AI/KF/readme.md and Plan.md
:: =============================================================================

python ./../Src/AI/KF/compare.py --make-new --r0-scale 1.15 --r1-scale 1.15 --epochs 10 --replay-n 50
pause
