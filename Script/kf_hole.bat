@echo off
:: =============================================================================
:: Script\kf_hole.bat  -  Task B fill-hole compare: drop -10 C from the existing grid,
::                 train a hole fleet, then frozen / retrain / replay / finetune
::                 / scale. Do not regenerate Data/grid. Do not touch Data/ai_mlp.
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_hole.bat
::   working directory is forced to the repo root
::
:: prerequisites
::   Data/grid/   existing 10x10 (copy only; never gen_grid into this folder)
::
:: command actually run
::   python Src/AI/KF/hole.py --epochs 10 --mlp-epochs 100 --replay-n 50
::
:: notes
::   copies  Data/grid_wo_tm10/     ~90 traces, no -10 C
::           Data/grid_tm10/        ~10 traces, only -10 C
::   trains  Data/ai_mlp_hole/      scheme B, new scaler (not Data/ai_mlp)
::   writes  Data/ai_kf/compare_hole/   compare.md / verdict.md / per-mode weights
::
:: resume hole fleet if val RMSE is still high:
::   python Src/AI/KF/hole.py --resume-mlp --skip-compare --mlp-epochs 200
:: then compare only:
::   python Src/AI/KF/hole.py --compare-only
::
:: split only / smoke:
::   python Src/AI/KF/hole.py --split-only
::   python Src/AI/KF/hole.py --smoke
::
:: acceptance (Doc/10 §7.1): Replay/retrain old-set worsen < 20%;
:: scale k ~ 1; finetune is the failure control.
:: =============================================================================

python ./../Src/AI/KF/hole.py --epochs 10 --mlp-epochs 100 --replay-n 50
pause
