@echo off
:: =============================================================================
:: Script\train_1000_resume.bat  -  continue scheme B for 1000 more voltage epochs
:: =============================================================================
::
:: usage
::   double-click this file, or from repo root:
::     Script\train_1000_resume.bat
::   working directory is forced to the repo root
::
:: prerequisites
::   Data/ai_mlp already has weights: last.pt or ckpts/epoch_XXXXX.pt
::   and scaler.json. Usually run train_100.bat first.
::
:: command actually run  -- edit the python line below
::   python Src/AI/MLP/train.py --scheme B --epochs 1000 --resume
::
:: args
::   --scheme B        must match the existing weights
::                     A: MLP outputs R0,R1,C1
::                     B: R0,R1 only, C1 fixed  -- default
::                     B+: R0,R1 plus one global C1
::   --epochs 1000     extra voltage epochs this run, NOT "train until 1000"
::   --resume          continue from the latest epoch  -- on in this file
::   --epoch N         continue from that epoch instead, e.g. --epoch 400
::                     do not combine with --resume
::   --ckpt PATH       continue from a file, e.g. Data/ai_mlp/best.pt
::   --fresh           ignore weights and train from scratch  -- do not add here
::   --pretrain-epochs N   teacher warmup; skipped on resume
::   --no-pretrain     disable warmup
::   --batch-size N    trajectory batch size  -- default 8
::   --lr X            learning rate  -- default 2e-3
::   --data-dir PATH   grid CSV dir  -- default Data/grid
::   --out-dir PATH    weights and logs  -- default Data/ai_mlp
::   --tbptt N         TBPTT window in steps; 0 = full trajectory
::   --device cpu/cuda  omit to auto-pick cuda if available
::   --list-ckpts      list saved epochs and exit
::
:: output  -- default Data/ai_mlp
::   each voltage epoch writes ckpts/epoch_XXXXX.pt and updates last.pt
::   best.pt if val RMSE improves
::   history.json is trimmed to the resume epoch
::
:: notes
::   --epochs is "how many more", not "train up to epoch N"
::   missing weights: run train_100.bat or check --out-dir
::   see Src/AI/MLP/readme.md
:: =============================================================================

python ./../Src/AI/MLP/train.py --scheme B --data-dir Data/grid_rc2 --out-dir Data/ai_mlp_rc2 --epochs 1000 --resume
pause
