@echo off
:: =============================================================================
:: train_100.bat  —  从头训练 MLP-ECM（方案 B，100 个电压 epoch）
:: =============================================================================
::
:: 用法
::   在仓库根目录双击，或命令行执行：
::     train_100.bat
::
:: 前置
::   先跑 gen_grid.bat（或 python Src/Sim/nmc100ah_ecm_gen_grid.py）
::   输出应在 Data/grid/（含 index.csv）。
::
:: 本文件实际执行的命令（只改下面这一行）
::   python Src/AI/MLP/train.py --scheme B --epochs 100
::
:: 参数说明
::   --scheme B        训练方案
::                       A   MLP 出 R0、R1、C1（激励很丰富再试）
::                       B   只出 R0、R1，C1 固定为 2.8e4 F（默认，先跑这个）
::                       B+  只出 R0、R1，再学一个全局 C1 标量
::   --epochs 100      电压训练轮数（不含预热；默认 40）
::   --pretrain-epochs N   教师 R0/R1 预热轮数（默认 5；0 跳过）
::   --no-pretrain     关掉预热
::   --batch-size N    轨迹批大小（默认 8）
::   --lr X            学习率（默认 2e-3）
::   --data-dir 路径   网格 CSV 目录（默认 Data/grid）
::   --out-dir 路径    权重与日志目录（默认 Data/ai_mlp）
::   --tbptt N         截断 BPTT 窗口步数；0 表示整条轨迹反传
::   --device cpu|cuda 不写则有 GPU 自动用 cuda
::   --resume          从最新 epoch 接着训（本文件不用；见 train_1000_resume.bat）
::   --epoch N         从指定电压 epoch 接着训，例如 --epoch 12
::   --ckpt 路径       直接指定权重文件
::   --fresh           忽略已有权重，强制从头训
::   --list-ckpts      只列出已保存的 epoch，然后退出
::
:: 输出（默认 Data/ai_mlp/）
::   best.pt  last.pt  ckpts/epoch_XXXXX.pt
::   config.json  scaler.json  history.json
::
:: 说明
::   本文件默认从头训。目录里已有权重时仍会重来，除非自行加上 --resume。
::   scaler.json 必须与权重配套，不要单独换。
::   详见 Src/AI/MLP/readme.md
:: =============================================================================

python Src/AI/MLP/train.py --scheme B --epochs 100
pause
