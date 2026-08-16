# Src/AI/MLP — 用电压误差训练 ECM 参数网络

MLP 输入电流、SOC、温度，输出 \(R_0,R_1\)（方案 A 再加 \(C_1\)），再经可微一阶 ECM 得到 \(\hat{U}_t\)。损失是 \(\hat{U}_t-U_t^{\mathrm{meas}}\)，梯度穿过电路回到网络。

设计见：

- `Doc/03-MLP-ECM物理信息参数估计.md`（方案 A）
- `Doc/04-MLP-ECM固定C1方案与对比.md`（方案 B / B+）
- `Doc/05-MLP-ECM增量学习方案与问题.md`（续训 ≠ 增量，方案与坑）
- `Doc/06-卡尔曼SOC与MLP-ECM融合增量学习.md`（EKF 估 SOC，开环电压误差做增量）
- `Doc/07-英飞凌残差头增量学习方案评估.md`（每芯 \(\Delta R\) 头，对照缩放 / Replay）
- `Doc/08-TC4D7-PPU与800V系统融合评估.md`（芯片 + 800 V 包 + demo / 优化）
- `Doc/09-非MLP结构GRU-LSTM评估.md`（GRU 不宜换 MLP；单点待久忘曲面是权重干涉）

闭环滤波和离线增量在 [`Src/AI/KF/`](../KF/readme.md)，不要在本目录的 `test.py` 里更新权重。

请在**仓库根目录**运行。依赖：`numpy`、`torch`（可选 `matplotlib` 做推理图）。

## 文件

| 文件 | 作用 |
|------|------|
| `config.py` | 方案、学习率、\(C_1^\star\)、数据路径 |
| `model.py` | `ParamMLP`，softplus 保证电阻电容为正 |
| `ecm.py` | 与 `nmc100ah_ecm_gen.py` 相同的离散化，可反传 |
| `dataset.py` | 读 `Data/grid/*.csv` |
| `train.py` | 预热 + 电压训练 |
| `infer.py` | 单条轨迹推理、可选画图 |
| `test.py` | 用 `nmc100ah_ecm_sim.csv` 对照电压与 \(R_0,R_1\) |

## 三种方案

| `--scheme` | MLP 输出 | \(C_1\) | 建议 |
|------------|----------|---------|------|
| `B`（默认） | \(R_0,R_1\) | 固定 `c1_star=2.8e4` F | 先跑这个 |
| `B+` | \(R_0,R_1\) | 再学一个全局标量 | 低温回弹不够时 |
| `A` | \(R_0,R_1,C_1\) | 逐步变化 | 激励很丰富再试 |

## 训练

先准备网格数据：

```powershell
python Src/Sim/nmc100ah_ecm_gen_grid.py
python Src/AI/MLP/train.py
python Src/AI/MLP/train.py --scheme B --epochs 40
python Src/AI/MLP/train.py --resume --epochs 20
python Src/AI/MLP/train.py --epoch 12 --epochs 10
python Src/AI/MLP/train.py --list-ckpts
python Src/AI/MLP/train.py --fresh --epochs 40
python Src/AI/MLP/train.py --scheme A --out-dir Data/ai_mlp_A
```

**默认是从头训。** 每个电压 epoch 会另存 `Data/ai_mlp/ckpts/epoch_00012.pt`，同时更新 `last.pt` / `best.pt`。

| 续训 | 起点 |
|------|------|
| `--resume` | 最新一个 epoch |
| `--epoch 12` | `epoch_00012.pt` |
| `--ckpt 路径` | 指定文件 |
| `--fresh` | 忽略已有，强制重来 |
| `--list-ckpts` | 只列出已保存的轮次 |

从中间某轮接着训时，history 会裁到该轮，避免和后面已经作废的记录缠在一起。归一化 `scaler.json` 必须保留。

默认流程：

1. 用 CSV 里教师 \(R_0,R_1\) 预热若干 epoch（`--no-pretrain` 可关）
2. 只用测量电压训练，ECM 状态 \(U_p\) 留在计算图里（整段 BPTT；`--tbptt N` 可截断）

权重写到 `Data/ai_mlp/best.pt`，另有 `config.json`、`scaler.json`、`history.json`。`Data/` 已 gitignore。

主损失不用教师参数列，那些列只作预热和事后对照。

## 推理

```powershell
python Src/AI/MLP/infer.py
python Src/AI/MLP/infer.py --epoch 12
python Src/AI/MLP/infer.py --best
python Src/AI/MLP/infer.py --csv Data/grid/nmc100ah_ecm_s02_t02_soc050_T+20.csv
```

推理默认用**最新 epoch**；`--best` 用验证集最好的那份。

输出 `Data/ai_mlp/infer.csv` 和 `Fig/mlp_ecm_infer.png`。

## 测试

默认拿 `Data/nmc100ah_ecm_sim.csv`（先跑 `nmc100ah_ecm_gen.py`）对照测量电压和教师 \(R_0,R_1\)：

```powershell
python Src/AI/MLP/test.py
python Src/AI/MLP/test.py --epoch 400
python Src/AI/MLP/test.py --best --show
```

默认用**最新 epoch**。四路纵排：电压测量/预测、电压误差、\(R_0\)、\(R_1\)。写出 `Data/ai_mlp/test.csv` 和 `Fig/mlp_ecm_test.png`。

## 代码里调用

```python
import sys
sys.path.insert(0, "Src/AI")
from MLP.config import TrainConfig
from MLP.model import ParamMLP
from MLP.ecm import ecm_forward

cfg = TrainConfig(scheme="B")
model = ParamMLP(cfg)
# x_norm: (B, T, 3) = 标准化后的 [I, SOC, T]
r0, r1, c1 = model(x_norm)
u_t, u_p = ecm_forward(i, u_ocv, r0, r1, c1, dt_s=0.1)
```

输入按训练集均值方差标准化，推理必须用同一次训练保存的 `scaler.json`。
