@echo off
:: =============================================================================
:: train_1000_resume.bat  —  从已有权重接着训（方案 B，再跑 1000 个电压 epoch）
:: =============================================================================
::
:: 用法
::   在仓库根目录双击，或命令行执行：
::     train_1000_resume.bat
::
:: 前置
::   Data/ai_mlp/ 下已有权重（last.pt / ckpts/epoch_XXXXX.pt）和 scaler.json。
::   一般先跑过 train_100.bat，再用来拉长训练。
::
:: 本文件实际执行的命令（只改下面这一行）
::   python Src/AI/MLP/train.py --scheme B --epochs 1000 --resume
::
:: 参数说明
::   --scheme B        训练方案，须与已有权重一致
::                       A   MLP 出 R0、R1、C1
::                       B   只出 R0、R1，C1 固定（默认）
::                       B+  只出 R0、R1，再学一个全局 C1
::   --epochs 1000     本次再训的电压轮数（不是累计到第 1000 轮）
::   --resume          从最新一个 epoch 接着训（本文件已打开）
::   --epoch N         改从指定轮次接着训，例如 --epoch 400
::                       与 --resume 二选一：写了 --epoch 就不要再写 --resume
::   --ckpt 路径       直接指定权重文件，例如 Data/ai_mlp/best.pt
::   --fresh           忽略已有权重，强制从头训（本文件不要加）
::   --pretrain-epochs N   教师 R0/R1 预热轮数（续训时一般不再预热）
::   --no-pretrain     关掉预热
::   --batch-size N    轨迹批大小（默认 8）
::   --lr X            学习率（默认 2e-3）
::   --data-dir 路径   网格 CSV 目录（默认 Data/grid）
::   --out-dir 路径    权重与日志目录（默认 Data/ai_mlp）
::   --tbptt N         截断 BPTT 窗口步数；0 表示整条轨迹反传
::   --device cpu|cuda 不写则有 GPU 自动用 cuda
::   --list-ckpts      只列出已保存的 epoch，然后退出
::
:: 输出（默认 Data/ai_mlp/）
::   每个电压 epoch 另存 ckpts/epoch_XXXXX.pt，并更新 last.pt / best.pt。
::   history.json 会接到续训起点，避免和已作废的后面记录缠在一起。
::
:: 说明
::   --epochs 是「再跑多少轮」，不是「训到第几轮」。
::   找不到可续训的权重时会报错，请先跑 train_100.bat 或检查 --out-dir。
::   详见 Src/AI/MLP/readme.md
:: =============================================================================

python Src/AI/MLP/train.py --scheme B --epochs 1000 --resume
pause
