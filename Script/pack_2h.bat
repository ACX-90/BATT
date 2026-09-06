@echo off
REM 第 2 期 2H1：AFE 停放（I_meas=0，I_cell=12 mA）。正式 n=180 / 48 h。
cd /d "%~dp0.."
if "%1"=="smoke" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2h1 --n 8 --seed 210 --engine ecm --park-h 6 --out-dir Data/pack/2h1_n8
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h1_n8 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h1_n8_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2h smoke done
  goto :eof
)
if "%1"=="2h2" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2h2 --n 180 --seed 211 --engine ecm --park-h 48 --out-dir Data/pack/2h2
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h2 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h2_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2h2 done
  goto :eof
)
python Src/Sim/nmc100ah_gen_pack.py --exp 2h1 --n 180 --seed 210 --engine ecm --park-h 48 --out-dir Data/pack/2h1
if errorlevel 1 exit /b 1
python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h1 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h1_kgrid
if errorlevel 1 exit /b 1
echo pack_2h done
