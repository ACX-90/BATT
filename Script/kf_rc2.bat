@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_rc2.bat  -  Task H: 2RC q=1 truth, BMS still 1RC. Filter only.
:: =============================================================================
::
:: does not train increment. writes Data/rc2/common.csv and Fig/kf_rc2.png
:: =============================================================================

python Src/Sim/nmc100ah_ecm_gen.py --out Data/rc2/common.csv --rc2
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --csv Data/rc2/common.csv --out Data/ai_kf/rc2_filter.csv --fig Fig/kf_rc2.png
:done
pause
