@echo off
REM 第 2 期 2G：全包 hatQ=0.95Qn 容量错（cc_rest，b_I=0）。正式 n=180。
cd /d "%~dp0.."
if "%1"=="smoke" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2g --n 8 --seed 209 --engine ecm --out-dir Data/pack/2g_n8
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2g_n8 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2g_n8_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2g smoke done
  goto :eof
)
python Src/Sim/nmc100ah_gen_pack.py --exp 2g --n 180 --seed 209 --engine ecm --out-dir Data/pack/2g
if errorlevel 1 exit /b 1
python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2g --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2g_kgrid
if errorlevel 1 exit /b 1
echo pack_2g done
