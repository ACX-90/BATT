@echo off
REM 第 2 期 2I：斜坡 / 震荡 / 驾驶电流（06-a §5.9）。不改 SEQUENCE 头。
cd /d "%~dp0.."
if "%1"=="smoke" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2i1 --n 8 --seed 213 --engine ecm --out-dir Data/pack/2i1_n8
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2i1_n8 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2i1_n8_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2i smoke 2i1 done
  goto :eof
)
if "%1"=="2i1" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2i1 --n 180 --seed 213 --engine ecm --out-dir Data/pack/2i1
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2i1 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2i1_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2i1 done
  goto :eof
)
if "%1"=="2i2" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2i2 --n 180 --seed 214 --engine ecm --out-dir Data/pack/2i2
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2i2 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2i2_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2i2 done
  goto :eof
)
if "%1"=="2i3" (
  python Src/Sim/nmc100ah_gen_pack.py --exp 2i3 --n 180 --seed 215 --engine ecm --out-dir Data/pack/2i3
  if errorlevel 1 exit /b 1
  python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2i3 --mode kgrid --mlp-dir Data/ai_mlp --out-dir Data/pack/2i3_kgrid
  if errorlevel 1 exit /b 1
  echo pack_2i3 done
  goto :eof
)
echo usage: pack_2i.bat [smoke^|2i1^|2i2^|2i3]
exit /b 1
