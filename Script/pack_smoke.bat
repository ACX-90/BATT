@echo off
REM 第 2 期烟测：n=8 的 2A1。数字作废，只证明脚本能跑、不读旧网格。
cd /d "%~dp0.."
python Src/AI/KF/pack_gate.py
if errorlevel 1 exit /b 1
python Src/Sim/nmc100ah_gen_pack.py --exp 2a1 --n 8 --engine pybamm --out-dir Data/pack/2a1_n8 --seed 201
if errorlevel 1 exit /b 1
python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1_n8 --mode freeze --out-dir Data/pack/2a1_n8_freeze
if errorlevel 1 exit /b 1
python Src/AI/EV_Local/pack.py --pack-dir Data/pack/2a1_n8 --mode kgrid --out-dir Data/pack/2a1_n8_kgrid
if errorlevel 1 exit /b 1
echo pack_smoke done
