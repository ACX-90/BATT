@echo off
:: =============================================================================
:: test.bat  —  用单条仿真轨迹测试已训 MLP-ECM
:: =============================================================================
::
:: 用法
::   在仓库根目录双击，或命令行执行：
::     test.bat
::
:: 前置
::   1. Data/ai_mlp/ 下已有权重和 scaler.json（先训练）
::   2. Data/nmc100ah_ecm_sim.csv 已生成（双击 gen_common.bat，
::      或 python Src/Sim/nmc100ah_ecm_gen.py）
::
:: 本文件实际执行的命令（只改下面这一行）
::   python Src/AI/MLP/test.py --show
::
:: 参数说明
::   --show            出图后弹窗（本文件已打开；不要图窗就删掉这个开关）
::   --no-plot         只算指标、写 CSV，不出图
::   --epoch N         用指定电压 epoch 的权重，例如 --epoch 400
::   --ckpt 路径       直接指定权重文件，例如 Data/ai_mlp/best.pt
::   --best            改用验证集最好的 best.pt（默认是最新 epoch）
::   --csv 路径        测试用仿真表（默认 Data/nmc100ah_ecm_sim.csv）
::   --out-dir 路径    找权重 / scaler 的目录（默认 Data/ai_mlp）
::   --out 路径        写出对照表（默认 Data/ai_mlp/test.csv）
::   --fig 路径        出图路径（默认 Fig/mlp_ecm_test.png）
::   --list-ckpts      只列出已保存的 epoch，然后退出
::
:: 默认行为
::   不写 --epoch / --ckpt / --best 时，用最新一个 epoch_XXXXX.pt。
::
:: 输出
::   控制台：电压 RMSE/MAE/MAX，以及 R0、R1 相对真值的误差
::   Data/ai_mlp/test.csv
::   Fig/mlp_ecm_test.png
::     四路纵排：测量/预测电压、电压误差、R0 真值/预测、R1 真值/预测
::
:: 说明
::   这是对照测试，不是训练。输入轨迹应带教师列 r0_ohm、r1_ohm。
::   详见 Src/AI/MLP/readme.md
:: =============================================================================

python Src/AI/MLP/test.py --show
pause
