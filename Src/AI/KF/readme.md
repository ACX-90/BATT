# Src/AI/KF — EKF 估 SOC + MLP–ECM 端电压 + 离线增量

状态只含 \((s,U_p)\)。MLP 仍只出 \((R_0,R_1)\)（方案 B 下 \(C_1\) 钉死），ECM 算 \(\hat U_t\)。卡尔曼每拍用电压创新改正 SOC 和极化，**不改 MLP 权重**。

设计见 `Doc/03-c-卡尔曼SOC与MLP-ECM融合增量学习.md`、`Doc/03-a-MLP-ECM增量学习方案与问题.md`。

请在**仓库根目录**运行。依赖：`numpy`、`torch`（出图再加 `matplotlib`）。先有 `Data/ai_mlp/` 的权重和 `scaler.json`。

## 一拍顺序

```
已有 s⁺, Up⁺
读入 I, T, Ut_meas
1. 安时预测          s⁻
2. MLP(I, s⁻, T)     → R0, R1    （C1 = C1★）
3. 极化预测          Up⁻
4. 先验电压          Ût⁻ = OCV(s⁻,T) − I R0 − Up⁻
5. 创新              e_pri = Ut_meas − Ût⁻
6. EKF 更新          s⁺, Up⁺
```

本拍电阻锁在预测 SOC 上，不用后验再算一遍。

## 三种电压误差

| 残差 | 含义 | 能否训 MLP |
|------|------|------------|
| \(e^{ol}\) | 纯安时 SOC + 同一套 ECM，不经过 KF | **增量主损失** |
| \(e^{pri}\) | 滤波创新 | 看 NIS，不用来反传 |
| \(e^{post}\) | 更新后残差 | **不要**当损失 |

## 文件

| 文件 | 作用 |
|------|------|
| `ocv.py` | 与仿真相同的 OCV 表、\(\partial U_{\mathrm{ocv}}/\partial s\)、反查 |
| `ekf.py` | 二维 EKF，可选慢变 \(\delta R_0\) |
| `adapter.py` | MLP 逐步出参；缩放适配 \(k_0,k_1\) |
| `filter.py` | 闭环滤波 + 开环对照 |
| `gate.py` | 增量门控 |
| `run.py` | 单条轨迹闭环 |
| `increment.py` | 离线增量：Replay / 缩放 / 只微调 / 合集重训 |
| `compare.py` | 四档对照：冻结 / 重训 / Replay / 微调 / 缩放 |
| `hole.py` | 任务 B：挖掉温区、另训洞舰队、再跑四档 |

## 闭环滤波

```powershell
python Src/AI/KF/run.py --selftest
python Src/AI/KF/run.py
python Src/AI/KF/run.py --soc-error 0.05 --current-bias 5
python Src/AI/KF/run.py --best --export Data/ai_kf/logs/sim.csv
python Src/AI/KF/run.py --r0-scale 1.2
```

| 参数 | 含义 |
|------|------|
| `--soc-error` | 开机 SOC 偏差 |
| `--current-bias` | 电流零偏 / A（放电为正），安时会漂，EKF 应用电压拉回来 |
| `--capacity-scale` | 容量用错的倍数 |
| `--r0-scale` | 故意放大 MLP 的 \(R_0\)，边沿 \(e^{ol}\) 应变大，SOC 不应被长期拉走 |
| `--export` | 再写一份带 `soc_ah` / `e_ol` 的日志，给增量用 |
| `--dr0` | 打开慢变 \(\delta R_0\)（默认关） |

输出：`Data/ai_kf/filter.csv`、`Fig/kf_ecm_filter.png`。

现场只给测量列 \(I,T,U_t\)。合成对照若 CSV 里有 `soc_true`，会打印安时 / EKF 的 SOC RMSE。

## 离线增量

在线只记账。过门控后再组轨迹，冻住旧 `scaler.json`，**禁止** `fit_scaler`，**禁止**预热，损失只用开环电压。

```powershell
python Src/AI/KF/increment.py --mode replay --new-dir Data/ai_kf/logs --replay-dir Data/grid --epochs 10
python Src/AI/KF/increment.py --mode scale --new-dir Data/ai_kf/logs --epochs 20 --lr 1e-2
python Src/AI/KF/increment.py --mode finetune --new-dir Data/ai_kf/logs
python Src/AI/KF/increment.py --eval-only --new-dir Data/grid --new-glob *T+50.csv
```

| `--mode` | 行为 |
|----------|------|
| `replay`（默认） | 新年份 + 旧网格回放混批，一期首选 |
| `scale` | 冻 MLP，只学 \(k_0,k_1\)，适合同一只电芯涨阻 |
| `finetune` | 只扫新数据，旧温区容易忘 |
| `retrain` | 旧+新合集，从旧权重接着训（冻 scaler）。对照上界，不是增量 |

电阻整张 ×1.15 冒充老化，一次跑四档（外加冻结基线）：

```powershell
python Src/AI/KF/compare.py --make-new --r0-scale 1.15 --r1-scale 1.15 --epochs 10
python Src/AI/KF/compare.py --smoke
```

`--make-new` 把缩放网格写到 `--new-dir`（默认 `Data/soh_k115/`），**不碰** `Data/grid/`。结果在 `Data/ai_kf/compare/compare.md`。读数见 [`Doc/04-a`](../../Doc/04-a-合成增量对照实验.md)，用法见 [`Doc/04-b`](../../Doc/04-b-增量学习应用手册.md)。整体涨阻时 `scale` 不该明显输给 `retrain`；`finetune` 旧集变差是预期失败对照。

填洞（任务 B，挖掉 −10 °C）走 `hole.py`，不要对全网格舰队做增量：

```powershell
python Src/AI/KF/hole.py
python Src/AI/KF/hole.py --split-only
python Src/AI/KF/hole.py --compare-only
```

写出 `Data/grid_wo_tm10/`、`Data/grid_tm10/`、`Data/ai_mlp_hole/`、`Data/ai_kf/compare_hole/`。`--task hole` 只改 `compare.md` 的验收口径。读数见 [`Doc/04-a`](../../Doc/04-a-合成增量对照实验.md) §7.1：Replay / 重训为正途，缩放 \(k\) 会拆开。

测量列舰队（任务 D）不要改 `config.py` 默认值：

```powershell
python Src/AI/MLP/train.py --scheme B --epochs 100 --out-dir Data/ai_mlp_meas --use-meas-inputs
python Src/AI/KF/compare.py --task meas --mlp-dir Data/ai_mlp_meas --new-dir Data/soh_k115 --old-dir Data/grid --out-dir Data/ai_kf/compare_meas
```

读数见 `Doc/04-a` §7.3：底板大约 +3 mV，×1.15 仍走缩放。

\(\delta R_0\)（任务 E）同一条 BOL 波，只把 MLP 的 \(R_0\) ×1.2：

```powershell
python Src/AI/KF/run.py --best --r0-scale 1.2 --out Data/ai_kf/dr0_off.csv
python Src/AI/KF/run.py --best --r0-scale 1.2 --dr0 --out Data/ai_kf/dr0_on.csv
```

读数见 `Doc/04-a` §7.4：开环仍大；打开后 SOC / NIS 回到默认，不要因此改 MLP。

小时级负例（任务 F）不要改默认 `SEQUENCE`：

```powershell
python Src/Sim/nmc100ah_ecm_gen_long.py
python Src/AI/KF/run.py --best --csv Data/long/cc_rest.csv
```

读数见 `Doc/04` §7.5。健康长恒流门控拒；零偏 / 容量错 NIS 仍健康，过了也不许拆 \(R\)。

权重写到 `Data/ai_kf/incr/`，**不覆盖** `Data/ai_mlp/best.pt`。滤波改用新表：

```powershell
python Src/AI/KF/run.py --mlp-dir Data/ai_kf/incr --best
```

验收：旧回放开环 RMSE 不明显变差；新轨迹开环 RMSE 下降；参考点（50%、25 °C、1C）的 \(R_0,R_1\) 漂移可解释。

## 一期不做

KF 状态里塞 MLP 权重、每拍 `backward`、增量方案 A、用 \(e^{post}\) 当 \(L_v\)、平台区无激励时更新 \(R\)、静默重拟合 scaler。
