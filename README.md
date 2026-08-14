# BATT — 100 Ah NMC 一阶 ECM 与物理信息参数估计

通用 **100 Ah NMC** 电芯的一阶 Thevenin 工具链：解析参数模型、时域仿真、出图，以及用端电压训练 MLP 估 \(R_0,R_1(,C_1)\)。

默认数值是模板，用来搭模型和跑通流程，**不是某一款商用电芯的出厂值**。换真实电芯时改参数，不改电路结构。

```
(I, T, SOC) ──► 解析 ECM / MLP ──► (R0, R1, C1) ──► 一阶电路 ──► Ut
                                                     │
                                      Ut_meas ───────┴──► 电压损失反传（仅 MLP）
```

请在**仓库根目录**运行脚本，相对路径 `Data/`、`Doc/`、`Fig/` 才正确。

## 目录

| 路径 | 作用 |
|------|------|
| [`Src/Sim/`](Src/Sim/readme.md) | 参数映射、单条仿真、SOC×温度网格出波 |
| [`Src/Plot/`](Src/Plot/readme.md) | \(R_0/R_1/C_1\) 曲面与仿真波形 |
| [`Src/AI/MLP/`](Src/AI/MLP/readme.md) | 物理信息 MLP：电压误差穿过可微 ECM |
| [`Doc/`](Doc/) | 参数规范、特性说明、MLP 方案推导 |
| `Data/` | 仿真 CSV、网格数据、训练产物（默认 gitignore） |
| `Fig/` | 出图 PNG（gitignore） |

## 环境

Python 3.10+。仿真只需 `numpy`；出图加 `matplotlib`；训练再加 `torch`。

```powershell
pip install numpy matplotlib torch
```

有 CUDA 时 `train.py` 会自动用 GPU，也可用 `--device cpu` 强制 CPU。

## 快速开始

### 1. 复核参数模型

```powershell
python Src/Sim/nmc100ah_ecm_demo.py
```

参考点（SOC = 50%，25 °C，1C 放电）：\(R_0=0.80\,\mathrm{m}\Omega\)，\(R_1=0.65\,\mathrm{m}\Omega\)，\(C_1=28000\,\mathrm{F}\)。

### 2. 单条工况仿真 + 出图

```powershell
python Src/Sim/nmc100ah_ecm_gen.py
python Src/Plot/plot_all.py
```

或双击 `gen_common.bat` 只出 CSV。

| 输出 | 内容 |
|------|------|
| `Data/nmc100ah_ecm_sim.csv` | 时域波形（`#` 头 + 表） |
| `Fig/nmc100ah_ecm_surfaces.png` | 参数曲面 |
| `Fig/nmc100ah_ecm_waveforms.png` | 电流 / 电压 / SOC / 极化 / \(R,C,\tau\) |

指令序列、噪声、初值改 `nmc100ah_ecm_gen.py` 文件头部。`--show` 弹窗，`--no-noise` 关噪声。

### 3. 网格数据 → 训练 MLP → 推理

```powershell
python Src/Sim/nmc100ah_ecm_gen_grid.py
python Src/AI/MLP/train.py --scheme B --epochs 40
python Src/AI/MLP/infer.py --best
```

Windows 也可按 `gen_grid.bat` → `train_100.bat` → `gen_common.bat` → `test.bat` 双击，见下面「批处理」。

默认方案 **B**：MLP 只出 \(R_0,R_1\)，\(C_1\) 钉在 \(2.8\times10^4\,\mathrm{F}\)。权重写到 `Data/ai_mlp/`，推理图 `Fig/mlp_ecm_infer.png`。

## 电路与约定

一阶 Thevenin（放电电流为正）：

\[
\dot{U}_p = -\frac{U_p}{R_1 C_1} + \frac{I}{C_1},\qquad
U_t = U_{\mathrm{ocv}}(\mathrm{SOC},T) - I R_0 - U_p
\]

| 项目 | 约定 |
|------|------|
| 电芯 | 100 Ah NMC，\(1\,\mathrm{C}=100\,\mathrm{A}\)，窗口 2.80 / 3.67 / 4.20 V |
| 电流 | 放电 \(>0\)，充电 \(<0\)，静置 \(=0\) |
| SOC | 默认 0~1；明显大于 1 时按百分数 |
| 温度 | °C（Arrhenius 内部转开尔文） |
| 电阻 / 电容 | \(\Omega\) / \(\mathrm{F}\)（图里常画成 mΩ、kF） |
| 仿真步长 | \(0.1\,\mathrm{s}\) |

解析映射是乘性因子，参考点处各因子为 1：

```
P = P_ref · f_SOC · f_相变 · f_T · f_I · f_充放
```

公式和默认系数见 [`Doc/NMC100Ah_ECM参数规范.md`](Doc/NMC100Ah_ECM参数规范.md)。

## 模块一览

### Sim — 仿真

| 脚本 | 作用 |
|------|------|
| `nmc100ah_ecm.py` | `NMC100AhECM.evaluate(I, T, SOC)` |
| `nmc100ah_ecm_params.py` | 参数数据类，可读写 JSON |
| `nmc100ah_ecm_demo.py` | 参考点复核、可选导出查找表 |
| `nmc100ah_ecm_gen.py` | 按充/放/静置序列写一份 CSV |
| `nmc100ah_ecm_gen_grid.py` | 默认 5×5 起始 SOC × 温度，写 `Data/grid/` |

代码里调用：

```python
import sys
sys.path.insert(0, "Src/Sim")
from nmc100ah_ecm import NMC100AhECM

R0, R1, C1 = NMC100AhECM().evaluate(i_a=100.0, t_celsius=25.0, soc=0.5)
```

网格默认档位：SOC = 0.10 / 0.30 / 0.50 / 0.70 / 0.90，温度 = −10 / 5 / 20 / 35 / 50 °C。低 SOC + 低温可能提前截止，是保护逻辑。

### Plot — 出图

```powershell
python Src/Plot/plot_all.py --csv Data/nmc100ah_ecm_sim.csv --show
```

曲面由 `NMC100AhECM` 现场算网格，不读 CSV。波形脚本读带 `#` 头的仿真表。Windows 下中文优先用微软雅黑。

### MLP — 灰箱估计

MLP 输入标准化后的 \([I,\,\mathrm{SOC},\,T]\)，经 softplus 保证参数为正，再走与仿真相同的离散化得到 \(\hat{U}_t\)。主损失是测量电压，教师 \(R,C\) 列只作预热和对照。

| `--scheme` | 输出 | \(C_1\) | 建议 |
|------------|------|---------|------|
| `B`（默认） | \(R_0,R_1\) | 固定 `c1_star` | 先跑这个 |
| `B+` | \(R_0,R_1\) | 再学一个全局标量 | 低温回弹不够时 |
| `A` | \(R_0,R_1,C_1\) | 逐步变化 | 激励很丰富再试 |

训练默认从头开始。`--resume` 从最新 epoch 续训，`--epoch N` / `--ckpt` 指定起点，`--fresh` 强制重来。归一化 `scaler.json` 必须与权重配套。

## 批处理

都在仓库根目录。资源管理器里双击即可（工作目录就是仓库根）。每个文件开头用 `::` 写了用法和参数，**改实际命令那一行就行**，不必先去翻 Python 脚本。

| 文件 | 做什么 | 实际命令 |
|------|--------|----------|
| [`gen_common.bat`](gen_common.bat) | 单条对照工况，写出 `Data/nmc100ah_ecm_sim.csv` | `python Src/Sim/nmc100ah_ecm_gen.py` |
| [`gen_grid.bat`](gen_grid.bat) | SOC×温度网格 10×10，写出 `Data/grid/`（先清旧 CSV） | `python .\Src\Sim\nmc100ah_ecm_gen_grid.py --n-soc 10 --n-temp 10` |
| [`train_100.bat`](train_100.bat) | 方案 B，从头训 100 个电压 epoch | `python Src/AI/MLP/train.py --scheme B --epochs 100` |
| [`train_1000_resume.bat`](train_1000_resume.bat) | 方案 B，从最新权重再训 1000 轮 | `python Src/AI/MLP/train.py --scheme B --epochs 1000 --resume` |
| [`test.bat`](test.bat) | 最新权重 + `Data/nmc100ah_ecm_sim.csv`，弹窗出对照图 | `python Src/AI/MLP/test.py --show` |
| [`autogit.bat`](autogit.bat) | `pull` → `add *` → `commit` → `push` | 见文件内四行 git |

常用改法（只动 bat 里那一行 python）：

| 目的 | 在对应 bat 里改成 |
|------|-------------------|
| 网格改回 5×5 | `gen_grid.bat` 里两个 `10` 都改成 `5` |
| 单条仿真关噪声 | `gen_common.bat` 加上 `--no-noise` |
| 训 200 轮 | `--epochs 200` |
| 换方案 A | `--scheme A` |
| 续训指定轮次 | 去掉 `--resume`，写成 `--epoch 400` |
| 测试指定权重 | `python Src/AI/MLP/test.py --epoch 400 --show` |
| 测试不弹窗 | 去掉 `--show` |

建议顺序：`gen_grid.bat` → `train_100.bat`（或 `train_1000_resume.bat`）→ `gen_common.bat` → `test.bat`。

`gen_grid.bat` 正式跑之前会清掉 `Data/grid/` 里旧 CSV。工况序列和噪声改 `Src/Sim/nmc100ah_ecm_gen.py` 头部，网格和单条仿真共用。`train_1000_resume.bat` 的 `--epochs` 是**再跑多少轮**，不是训到第几轮。`test.bat` 默认用最新 `epoch_XXXXX.pt`；`--best` 改用验证集最好的那份。

## 文档

| 文档 | 内容 |
|------|------|
| [`Doc/NCM电芯ECM参数R0_R1_C1特性.md`](Doc/NCM电芯ECM参数R0_R1_C1特性.md) | \(R_0,R_1,C_1\) 随 SOC / 温度 / 电流的定性规律 |
| [`Doc/NMC100Ah_ECM参数规范.md`](Doc/NMC100Ah_ECM参数规范.md) | 乘性结构、默认系数、标定建议 |
| [`Doc/MLP-ECM物理信息参数估计.md`](Doc/MLP-ECM物理信息参数估计.md) | 方案 A：MLP 出三个参数，电压反传 |
| [`Doc/MLP-ECM固定C1方案与对比.md`](Doc/MLP-ECM固定C1方案与对比.md) | 方案 B / B+：为什么少出 \(C_1\) 更好训 |
| [`Doc/MLP-ECM增量学习方案与问题.md`](Doc/MLP-ECM增量学习方案与问题.md) | `--resume` 不是增量；回放 / 缩放 / 扩维及本仓库的坑 |

子目录 `readme.md` 写各自的命令、列名和配置项。
