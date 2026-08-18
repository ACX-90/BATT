@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_dr0.bat  -  Task E: same BOL waveform, MLP R0 x1.2, EKF --dr0 off/on
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_dr0.bat
::
:: command actually run
::   python Src/AI/KF/run.py --best --out Data/ai_kf/dr0_base.csv --fig Fig/kf_dr0_base.png
::   python Src/AI/KF/run.py --best --r0-scale 1.2 --out Data/ai_kf/dr0_off.csv --fig Fig/kf_dr0_off.png
::   python Src/AI/KF/run.py --best --r0-scale 1.2 --dr0 --out Data/ai_kf/dr0_on.csv --fig Fig/kf_dr0_on.png
::
:: notes
::   Data/nmc100ah_ecm_sim.csv must exist (Script\gen_common.bat)
::   uses Data/ai_mlp/best.pt (500+ epoch fleet). Does not touch MLP weights.
:: =============================================================================

python Src/AI/KF/run.py --best --out Data/ai_kf/dr0_base.csv --fig Fig/kf_dr0_base.png
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --r0-scale 1.2 --out Data/ai_kf/dr0_off.csv --fig Fig/kf_dr0_off.png
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --r0-scale 1.2 --dr0 --out Data/ai_kf/dr0_on.csv --fig Fig/kf_dr0_on.png
:done
pause
