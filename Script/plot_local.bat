@echo off
:: =============================================================================
:: Script\plot_local.bat  -  plot phase-1 open-loop residual (freeze vs 1a vs 1b)
:: =============================================================================
::
:: from repo root:
::   Script\plot_local.bat
::
:: prerequisites
::   1. Data/ai_mlp/best.pt
::   2. Data/soh_k115/*.csv
::   3. optional Data/ai_local/kgrid_k115_p4/last.pt  (1b overlay + heatmap)
::
:: output
::   Fig/local/phase1_mid_wave.png
::   Fig/local/phase1_resid_T.png
::   Fig/local/phase1_zoom.png
::   Fig/local/phase1_kgrid.png
::   Fig/local/phase1_soc_mid.png
::   Fig/local/phase1_soc_T.png
:: =============================================================================

cd /d "%~dp0\.."
python Src/AI/EV_Local/plot.py %*
