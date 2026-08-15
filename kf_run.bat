@echo off
:: =============================================================================
:: kf_run.bat  -  EKF SOC + MLP-ECM terminal voltage on one trajectory
:: =============================================================================
::
:: usage
::   double-click in repo root, or:
::     kf_run.bat
::
:: prerequisites
::   1. Data/ai_mlp weights + scaler.json
::   2. Data/nmc100ah_ecm_sim.csv   (gen_common.bat)
::
:: command actually run
::   python Src/AI/KF/run.py --soc-error 0.05 --current-bias 5 --show
::
:: notes
::   --selftest          EKF rest correction only
::   --soc-error 0.05    wrong initial SOC
::   --current-bias 5    5 A current bias; Ah drifts, EKF should pull SOC back
::   --export PATH       write incremental log with soc_ah / e_ol
::   --best              use Data/ai_mlp/best.pt
::   see Src/AI/KF/readme.md
:: =============================================================================

python Src/AI/KF/run.py --soc-error 0.05 --current-bias 5 --show
pause
