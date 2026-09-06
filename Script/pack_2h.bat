@echo off
REM 第 2 期 2H：AFE 停放。2H1 均匀 12 mA；2H2 daisy 顶芯 +2 mA；2H3 1C 充再停（§3.6）。
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
if "%1"=="2h3" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2h3 --n 180 --seed 212 --engine ecm --park-h 48 --charge-min 5 --out-dir Data/pack/2h3
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h3 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h3_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2h3 done
  goto :eof
)
if "%1"=="2h3-smoke" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2h3 --n 8 --seed 212 --engine ecm --park-h 1 --charge-min 5 --out-dir Data/pack/2h3_n8
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h3_n8 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h3_n8_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2h3 smoke done
  goto :eof
)
python Src/Sim/nmc100ah_gen_pack.py --exp 2h1 --n 180 --seed 210 --engine ecm --park-h 48 --out-dir Data/pack/2h1
if errorlevel 1 exit /b 1
python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2h1 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2h1_kgrid
if errorlevel 1 exit /b 1
echo pack_2h done
