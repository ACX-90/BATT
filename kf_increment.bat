@echo off
:: =============================================================================
:: kf_increment.bat  -  offline incremental MLP (open-loop voltage, frozen scaler)
:: =============================================================================
::
:: usage
::   double-click in repo root, or:
::     kf_increment.bat
::
:: prerequisites
::   1. trained Data/ai_mlp  (scaler.json must stay)
::   2. new logs in Data/ai_kf/logs   or use --new-dir Data/grid --new-glob ...
::   3. keep Data/grid for replay; do not re-run gen_grid on the same folder
::
:: command actually run
::   python Src/AI/KF/increment.py --mode replay --new-dir Data/ai_kf/logs --replay-dir Data/grid --epochs 10
::
:: notes
::   replay   mix new + old grid   (default, first choice)
::   scale    freeze MLP, learn k0 k1 only
::   finetune new data only, forgets old T/SOC
::   writes Data/ai_kf/incr , does not overwrite Data/ai_mlp/best.pt
::   see Src/AI/KF/readme.md
:: =============================================================================

python Src/AI/KF/increment.py --mode replay --new-dir Data/ai_kf/logs --replay-dir Data/grid --epochs 10
pause
