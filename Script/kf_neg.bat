@echo off
cd /d "%~dp0.."
:: =============================================================================
:: Script\kf_neg.bat  -  Task F: hour-scale negative traces + gate / filter
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\kf_neg.bat
::
:: does
::   1. generate Data/long/*.csv via nmc100ah_gen_long.py (does not edit SEQUENCE)
::   2. filter cc_rest, then same trace with --current-bias 5 and --capacity-scale 0.95
::   3. filter chg_park
::
:: notes
::   gate may PASS on pure CC+rest (edge or 30 s rest). bias / Q-error should fail NIS.
::   do not finetune the 2 h traces (full BPTT). this bat is run.py + gate only.
:: =============================================================================

python Src/Sim/nmc100ah_gen_long.py
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --csv Data/long/cc_rest.csv --export Data/ai_kf/logs/neg_cc.csv --out Data/ai_kf/neg_cc.csv --fig Fig/kf_neg_cc.png
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --csv Data/long/cc_rest.csv --current-bias 5 --export Data/ai_kf/logs/neg_bias.csv --out Data/ai_kf/neg_bias.csv --fig Fig/kf_neg_bias.png
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --csv Data/long/cc_rest.csv --capacity-scale 0.95 --export Data/ai_kf/logs/neg_q.csv --out Data/ai_kf/neg_q.csv --fig Fig/kf_neg_q.png
if errorlevel 1 goto :done
python Src/AI/KF/run.py --best --csv Data/long/chg_park.csv --export Data/ai_kf/logs/neg_park.csv --out Data/ai_kf/neg_park.csv --fig Fig/kf_neg_park.png
:done
pause
