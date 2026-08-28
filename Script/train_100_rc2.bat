@echo off
:: =============================================================================
:: Script\train_100.bat  -  train MLP-ECM from scratch, scheme B, 100 voltage epochs
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\train_100.bat
::   working directory is forced to the repo root
::
:: prerequisites
::   run gen_grid.bat first, or: python Src/Sim/nmc100ah_ecm_gen_grid.py
::   expect Data/grid with index.csv
::
:: command actually run  -- edit the python line below
::   python Src/AI/MLP/train.py --scheme B --epochs 100
::
:: args
::   --scheme B        A: MLP outputs R0,R1,C1
::                     B: R0,R1 only, C1 fixed at 2.8e4 F  -- default
::                     B+: R0,R1 plus one global C1 scalar
::   --epochs 100      voltage epochs, not counting pretrain  -- default 40
::   --pretrain-epochs N   teacher R0/R1 warmup  -- default 5; 0 skips
::   --no-pretrain     disable warmup
::   --batch-size N    trajectory batch size  -- default 8
::   --lr X            learning rate  -- default 2e-3
::   --data-dir PATH   grid CSV dir  -- default Data/grid
::   --out-dir PATH    weights and logs  -- default Data/ai_mlp
::   --tbptt N         TBPTT window in steps; 0 = full trajectory
::   --device cpu/cuda  omit to auto-pick cuda if available
::   --resume          continue from latest epoch  -- not used here
::   --epoch N         continue from that voltage epoch, e.g. --epoch 12
::   --ckpt PATH       continue from a weight file
::   --fresh           ignore existing weights, force scratch
::   --list-ckpts      list saved epochs and exit
::
:: output  -- default Data/ai_mlp
::   best.pt  last.pt  ckpts/epoch_XXXXX.pt
::   config.json  scaler.json  history.json
::
:: notes
::   this file trains from scratch even if weights already exist
::   add --resume to continue; see train_1000_resume.bat
::   keep scaler.json with the weights
::   see Src/AI/MLP/readme.md
:: =============================================================================

python ./../Src/AI/MLP/train.py --scheme B --data-dir Data/grid_rc2 --out-dir Data/ai_mlp_rc2 --no-pretrain
pause
