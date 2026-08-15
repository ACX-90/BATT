# Src/AI/KF — EKF 估 SOC + MLP–ECM 端电压 + 离线增量

状态只含 \((s,U_p)\)。MLP 仍只出 \((R_0,R_1)\)（方案 B 下 \(C_1\) 钉死），ECM 算 \(\hat U_t\)。卡尔曼每拍用电压创新改正 SOC 和极化，**不改 MLP 权重**。

设计见 `Doc/06-卡尔曼SOC与MLP-ECM融合增量学习.md`、`Doc/05-MLP-ECM增量学习方案与问题.md`。

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
| `increment.py` | 离线增量：Replay / 缩放 / 只微调 |

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

权重写到 `Data/ai_kf/incr/`，**不覆盖** `Data/ai_mlp/best.pt`。滤波改用新表：

```powershell
python Src/AI/KF/run.py --mlp-dir Data/ai_kf/incr --best
```

验收：旧回放开环 RMSE 不明显变差；新轨迹开环 RMSE 下降；参考点（50%、25 °C、1C）的 \(R_0,R_1\) 漂移可解释。

## 一期不做

KF 状态里塞 MLP 权重、每拍 `backward`、增量方案 A、用 \(e^{post}\) 当 \(L_v\)、平台区无激励时更新 \(R\)、静默重拟合 scaler。
