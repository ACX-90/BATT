@echo off
setlocal
cd /d "%~dp0\.."
if not exist Src\MCU_Eval\build mkdir Src\MCU_Eval\build
gcc -O2 -std=c99 -Wall -Isrc/MCU_Eval -ISrc/MCU_Eval ^
  Src/MCU_Eval/mcu_eval_common.c ^
  Src/MCU_Eval/eval_ecm.c ^
  Src/MCU_Eval/eval_ekf.c ^
  Src/MCU_Eval/eval_mlp64.c ^
  Src/MCU_Eval/eval_mlp16.c ^
  Src/MCU_Eval/eval_lut.c ^
  Src/MCU_Eval/eval_head3x8x2.c ^
  Src/MCU_Eval/eval_k_global.c ^
  Src/MCU_Eval/eval_k_grid.c ^
  Src/MCU_Eval/eval_pulse_r0.c ^
  Src/MCU_Eval/eval_incr_kgrid.c ^
  Src/MCU_Eval/eval_sox_cell.c ^
  Src/MCU_Eval/eval_sox_pack.c ^
  Src/MCU_Eval/eval_all.c ^
  -lm -o Src/MCU_Eval/build/mcu_eval_host.exe
if errorlevel 1 exit /b 1
Src\MCU_Eval\build\mcu_eval_host.exe
endlocal
