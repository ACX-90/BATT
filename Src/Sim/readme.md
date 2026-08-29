# Src/Sim — 100 Ah NMC 一阶 ECM 仿真

本目录是电芯等效电路模型的计算核心：参数映射 `(I, T, SOC) → (R0, R1, C1)`、单条工况时域仿真、SOC×温度网格批量出波。

更完整的公式与系数见仓库 `Doc/01-b-NMC100Ah_ECM参数规范.md`。

## 文件

| 文件 | 作用 |
|------|------|
| `nmc100ah_ecm_params.py` | 参数数据类与默认值，可读写 JSON |
| `nmc100ah_ecm.py` | `NMC100AhECM.evaluate`：算 R0 / R1 / C1 |
| `nmc100ah_ecm_demo.py` | 参考点复核、典型工况打印、可选导出查找表 |
| `nmc100ah_gen.py` | 时域仿真入口：默认 ECM，`--pybamm` 换 SPM |
| `nmc100ah_gen_grid.py` | 网格批量出波；默认 ECM，`--pybamm --out-dir Data/grid_pybamm` |
| `nmc100ah_gen_long.py` | 小时级负例；同样默认 ECM，`--pybamm` 写 `Data/long_pybamm/` |
| `nmc100ah_pybamm.py` | PyBaMM SPM 后端（Chen2020→100Ah），由上面三个入口的 `--pybamm` 调用 |

请在**仓库根目录**运行脚本，相对路径 `Data/`、`Doc/` 才正确。

依赖：`numpy`。`--pybamm` 另需 `pip install pybamm`。

## 约定

- 放电电流为正，充电为负，静置为 0
- SOC 默认 0~1；明显大于 1 时按百分数
- 温度单位 °C
- 电阻单位 Ω（图里常画成 mΩ），电容单位 F
- 仿真步长 `DT_S = 0.1` s
- 参考点：SOC = 50%，T = 25 °C，I = 100 A 放电  
  此时 `R0 = 0.80 mΩ`，`R1 = 0.65 mΩ`，`C1 = 28000 F`
- 默认数值是通用 100 Ah NMC 模板，不是某一款商用电芯的出厂值

## 参数模型

```powershell
python Src/Sim/nmc100ah_ecm_demo.py
python Src/Sim/nmc100ah_ecm_demo.py --json Doc/NMC100Ah_ECM_params.json --csv Doc/NMC100Ah_ECM_lookup.csv
```

代码里调用：

```python
import sys
sys.path.insert(0, "Src/Sim")
from nmc100ah_ecm import NMC100AhECM

model = NMC100AhECM()
R0, R1, C1 = model.evaluate(i_a=100.0, t_celsius=25.0, soc=0.5)
# 或：model.evaluate(i_c=1.0, t_celsius=25.0, soc=50, soc_in_percent=True)
```

结构：

```
P = P_ref · f_SOC · f_相变 · f_T · f_I · f_充放
```

改真实电芯时优先改 `ref_value` 和温度活化能，不要先改乘性结构。也可用 `ECMParamSet.to_json` / `NMC100AhECM.from_json`。

## 单条仿真

改 `nmc100ah_gen.py` 文件头部：

- `SOC0`、`T_AMBIENT_C`、`DT_S`
- `NOISE_ENABLE`、`NOISE_SEED`、`NOISE_STD`（只污染 `*_meas` 列）
- `SEQUENCE`：`charge` / `discharge` / `rest`

```python
SEQUENCE = [
    {"mode": "rest", "duration_s": 30.0},
    {"mode": "discharge", "duration_s": 180.0, "c_rate": 1.0},
    {"mode": "charge", "duration_s": 90.0, "c_rate": 0.5},
]
```

`c_rate` 与 `current_a` 二选一。碰到电压或 SOC 边界时，本条指令剩余时间改为静置。

```powershell
python Src/Sim/nmc100ah_gen.py
python Src/Sim/nmc100ah_gen.py --out Data/my_run.csv
python Src/Sim/nmc100ah_gen.py --no-noise
```

`Script/gen_common.bat` 等价于第一条。其它模块不要改本文件头部 `SEQUENCE`，传入可选工况：

```python
from nmc100ah_gen import run_sim
run_sim(out="Data/long/cc_rest.csv", sequence=my_seq, soc0=0.70)
```

小时级负例（任务 F）走 `nmc100ah_gen_long.py`，默认网格工况不变。`schg_park` 里的 `chg_ramp` / `dis_ramp` 按 `c_rate_start` → `c_rate_end` 逐步线性变电流。`--pybamm` 把同一条坡度做成 drive cycle，默认写到 `Data/long_pybamm/`，不覆盖 `Data/long/`。

```powershell
python Src/Sim/nmc100ah_gen_long.py --only schg_park
python Src/Sim/nmc100ah_gen_long.py --only schg_park --pybamm --no-noise
```

寿命因子与 2RC（仿真比 BMS 更真，BMS 仍读 1RC）：

```powershell
python Src/Sim/nmc100ah_gen_grid.py --out-dir Data/soh_q90 --soh 0.90
python Src/Sim/nmc100ah_gen.py --out Data/rc2/common.csv --rc2
```

`--soh 0.90` 只改 \(R,C\)（\(a_0=1,a_1=1.5,a_C=-0.3\)），OCV 和容量默认不动。`--rc2` 叠加 \(\tau_2=90\,\mathrm{s}\)、\(R_2\approx 0.43\,R_1\)。两个开关可分开。不要写回 `Data/grid/`。

默认写出 `Data/nmc100ah_ecm_sim.csv`。前几行是 `#` 元数据，pandas 读取时加 `comment="#"`。

主要列：`time_s, mode, i_true_a, i_meas_a, t_true_c, soc_true, u_ocv_v, r0_ohm, r1_ohm, c1_f, tau1_s, u_p_v, u_t_true_v, u_t_meas_v`。

## 网格批量仿真

`nmc100ah_gen_grid.py` 沿用上面的 `SEQUENCE` 和噪声配置，只改每份的起始 SOC 与温度。

默认 5×5 = 25 份：

| 维 | 默认档位 |
|----|----------|
| 起始 SOC | 0.10, 0.30, 0.50, 0.70, 0.90 |
| 温度 / °C | −10, 5, 20, 35, 50 |

改头部 `N_SOC`、`N_TEMP`、`SOC_MIN`/`SOC_MAX`、`T_MIN_C`/`T_MAX_C`，或直接写列表：

```python
SOC_VALUES = [0.15, 0.40, 0.70]
T_VALUES_C = [-10, 25, 45]
```

```powershell
python Src/Sim/nmc100ah_gen_grid.py
python Src/Sim/nmc100ah_gen_grid.py --n-soc 5 --n-temp 5
python Src/Sim/nmc100ah_gen_grid.py --dry-run
python Src/Sim/nmc100ah_gen_grid.py --out-dir Data/soh_k115 --r0-scale 1.15 --r1-scale 1.15
python Src/Sim/nmc100ah_gen_grid.py --out-dir Data/grid_noisy --n-soc 10 --n-temp 10 --noise-voltage 0.007 --noise-current 0.1 --noise-temp 0.5 --noise-soc 0.005
```

加大噪声写到**新目录**，不要改默认 `Data/grid/`。`--noise-voltage` 单位是伏特（0.007 = 7 mV）。

`--r0-scale` / `--r1-scale` 把解析 \(R\) 整张乘常数后再仿真，用来造换对象 / 假老化。必须换 `--out-dir`，不要写回 `Data/grid/`。

`Script/gen_grid.bat` 跑的是 `--n-soc 10 --n-temp 10`（100 份）。

可选 PyBaMM（不要写回默认 `Data/grid/`）：

```powershell
python Src/Sim/nmc100ah_pybamm.py
python Src/Sim/nmc100ah_gen.py --pybamm --out Data/nmc100ah_pybamm_sim.csv
python Src/Sim/nmc100ah_gen_grid.py --pybamm --out-dir Data/grid_pybamm
python Src/Sim/nmc100ah_gen_grid.py --pybamm --edge-c-half --out-dir Data/grid_pybamm
python Src/Sim/nmc100ah_gen_grid.py --pybamm --n-soc 5 --n-temp 5 --out-dir Data/grid_pybamm --dry-run
```

`--pybamm` 沿用同一套 `SEQUENCE`、噪声、步长和 CSV 列。端电压 / OCV / SOC 来自 SPM；`r0_ohm,r1_ohm,c1_f` 在同一 `(I,T,SOC)` 上用仓库 ECM 求值，只作教师列。默认等温，温度轴与 ECM 网格同义；`--thermal` 打开 lumped 热。不能和 `--rc2` / `--soh` / `--r0-scale` 叠用。`--edge-c-half` 只改该份边沿电流（SOC>0.85 充电减半，SOC<0.15 放电减半），不改默认 SEQUENCE。

正式跑之前会先删掉输出目录里已有的 `*.csv`（含 `index.csv`），避免换档数后旧文件混进训练集。`--dry-run` 不删、不写。

输出在 `Data/grid/`：

- `nmc100ah_ecm_s{ii}_t{jj}_socXXX_T±YY.csv`
- `index.csv`：档位、初值、结束 SOC/电压、是否触发保护、路径

低 SOC + 低温可能提前截止，属保护逻辑，不是算挂了。

## 输出目录

`Data/`、`Data/grid/` 已在仓库根目录 `.gitignore` 中，仿真结果不会进 git。
