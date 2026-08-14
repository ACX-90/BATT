@echo off
:: =============================================================================
:: gen_grid.bat  —  按起始 SOC × 温度网格批量生成训练波形
:: =============================================================================
::
:: 用法
::   在仓库根目录双击，或命令行执行：
::     gen_grid.bat
::
:: 做什么
::   每份沿用 nmc100ah_ecm_gen.py 的 SEQUENCE 和噪声，只改起始 SOC 与温度。
::   写出 Data/grid/*.csv，供 train_100.bat / train_1000_resume.bat 使用。
::
:: 本文件实际执行的命令（只改下面这一行）
::   python .\Src\Sim\nmc100ah_ecm_gen_grid.py --n-soc 10 --n-temp 10
::
:: 命令行参数
::   --n-soc N         起始 SOC 档数（本文件 10；脚本默认 5）
::   --n-temp N        温度档数（本文件 10；脚本默认 5）
::   --out-dir 路径    输出目录（默认 Data/grid）
::   --seed N          覆盖基础噪声种子；每档再偏置 i*100+j
::   --no-noise        关闭测量噪声
::   --dry-run         只打印网格，不删旧文件、不仿真
::
:: 扫描区间在 Src/Sim/nmc100ah_ecm_gen_grid.py 头部：
::   SOC_MIN / SOC_MAX     默认 0.10 ~ 0.90（含端点；n=1 取中点）
::   T_MIN_C / T_MAX_C     默认 -10 ~ 50 °C
::   SOC_VALUES / T_VALUES_C
::                         若写成列表则不再按 MIN/MAX 均分，档数以列表为准
::   OUTPUT_DIR / FILE_NAME
::
:: 指令序列、噪声、步长
::   全部从 nmc100ah_ecm_gen.py 导入，改那一处即可两边同步。
::
:: 输出（默认 Data/grid/）
::   nmc100ah_ecm_s{ii}_t{jj}_socXXX_T±YY.csv
::   index.csv   档位、初值、结束 SOC/电压、是否触发保护、路径
::
:: 说明
::   正式跑之前会先删掉输出目录里已有的 *.csv（含 index.csv），
::   避免换档数后旧文件混进训练集。--dry-run 不删、不写。
::   本文件是 10×10 = 100 份；改回 5×5 就把两个 10 都改成 5。
::   低 SOC + 低温可能提前截止，是保护逻辑，不是算挂了。
::   详见 Src/Sim/readme.md
:: =============================================================================

python .\Src\Sim\nmc100ah_ecm_gen_grid.py --n-soc 10 --n-temp 10
pause
