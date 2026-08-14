@echo off
:: =============================================================================
:: gen_common.bat  —  生成单条对照工况（Data/nmc100ah_ecm_sim.csv）
:: =============================================================================
::
:: 用法
::   在仓库根目录双击，或命令行执行：
::     gen_common.bat
::
:: 做什么
::   按 nmc100ah_ecm_gen.py 头部的 SEQUENCE 跑一条时域仿真，
::   给 test.bat / 出图用。训练网格请用 gen_grid.bat。
::
:: 本文件实际执行的命令（只改下面这一行）
::   python Src/Sim/nmc100ah_ecm_gen.py
::
:: 命令行参数
::   --out 路径        输出 CSV（默认 Data/nmc100ah_ecm_sim.csv）
::   --seed N          覆盖头部 NOISE_SEED
::   --no-noise        关闭测量噪声（真值列本来就没有噪声）
::
:: 工况本身不在 bat 里改，打开 Src/Sim/nmc100ah_ecm_gen.py 头部：
::   SOC0 / T_AMBIENT_C / U_P0    初始 SOC、温度、极化电压
::   DT_S                         步长，默认 0.1 s
::   ENABLE_CUTOFF                碰到电压/SOC 边界则本条指令改静置
::   NOISE_ENABLE / NOISE_SEED / NOISE_STD
::   SEQUENCE                     充 / 放 / 静置指令列表
::     mode          charge | discharge | rest
::     duration_s    秒（与 duration_steps 二选一）
::     c_rate        倍率，1.0 = 100 A（与 current_a 二选一）
::
:: 输出
::   Data/nmc100ah_ecm_sim.csv
::   前几行是 # 元数据，表头后才是数据。pandas 读时加 comment="#"。
::
:: 说明
::   放电电流为正，充电为负。默认数值是 100 Ah NMC 模板。
::   网格批量出波用 gen_grid.bat（沿用本文件的 SEQUENCE 和噪声配置）。
::   详见 Src/Sim/readme.md
:: =============================================================================

python Src/Sim/nmc100ah_ecm_gen.py
pause
