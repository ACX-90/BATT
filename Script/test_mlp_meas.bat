@echo off
:: =============================================================================
:: Script\test.bat  -  test trained MLP-ECM on one sim trajectory
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\test.bat
::   working directory is forced to the repo root
::
:: prerequisites
::   1. weights and scaler.json under Data/ai_mlp  -- train first
::   2. Data/nmc100ah_ecm_sim.csv exists
::      run gen_common.bat or: python Src/Sim/nmc100ah_ecm_gen.py
::
:: command actually run  -- edit the python line below
::   python Src/AI/MLP/test.py --show
::
:: args
::   --show            pop up the figure  -- on in this file; drop it to skip
::   --no-plot         metrics and CSV only, no figure
::   --epoch N         use that voltage-epoch ckpt, e.g. --epoch 400
::   --ckpt PATH       weight file, e.g. Data/ai_mlp/best.pt
::   --best            use best.pt instead of the latest epoch
::   --csv PATH        sim table  -- default Data/nmc100ah_ecm_sim.csv
::   --out-dir PATH    dir for weights and scaler  -- default Data/ai_mlp
::   --out PATH        write compare table  -- default Data/ai_mlp/test.csv
::   --fig PATH        figure path  -- default Fig/mlp_ecm_test.png
::   --list-ckpts      list saved epochs and exit
::
:: default
::   with no --epoch, --ckpt or --best: latest epoch_XXXXX.pt
::
:: output
::   console: voltage RMSE, MAE, MAX; R0 and R1 vs teacher
::   Data/ai_mlp/test.csv
::   Fig/mlp_ecm_test.png
::     4 rows: Ut meas/pred, voltage error, R0 true/pred, R1 true/pred
::
:: notes
::   evaluation only, no training, weights are not written back
::   CSV should have teacher columns r0_ohm, r1_ohm
::   see Src/AI/MLP/readme.md
:: =============================================================================

python ./../Src/AI/MLP/test.py --show --ckpt Data/ai_mlp_meas/best.pt
