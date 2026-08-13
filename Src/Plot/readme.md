# Src/Plot — ECM 参数曲面与仿真波形

本目录把 `Src/Sim` 的参数模型和 `Data/*.csv` 画成图，结果写到仓库根目录 `Fig/`。

请在**仓库根目录**运行。依赖：`numpy`、`matplotlib`。Windows 下中文优先用微软雅黑。

## 文件

| 文件 | 作用 |
|------|------|
| `plot_ecm_surfaces.py` | R0 / R1 / C1 三维曲面 |
| `plot_sim_waveforms.py` | 单份仿真 CSV 的时域波形 |
| `plot_all.py` | 一次出曲面 + 波形 |
| `_common.py` | 路径、字体、读带 `#` 头的 CSV、保存图片 |

## 一次出两张图

```powershell
python Src/Plot/plot_all.py
python Src/Plot/plot_all.py --show
python Src/Plot/plot_all.py --csv Data/nmc100ah_ecm_sim.csv
```

不加 `--show` 只存 PNG、不弹窗。

| 输出 | 内容 |
|------|------|
| `Fig/nmc100ah_ecm_surfaces.png` | 参数曲面 |
| `Fig/nmc100ah_ecm_waveforms.png` | 仿真波形 |

`Fig/` 已 gitignore，图片不会上传。

## 参数曲面

由 `NMC100AhECM` 现场算网格，不读 CSV。

```powershell
python Src/Plot/plot_ecm_surfaces.py
python Src/Plot/plot_ecm_surfaces.py --show
```

一张 2×3：

- 上排：固定 1C 放电（100 A），\(R_0,R_1,C_1\) 对 SOC–温度
- 下排：固定 25 °C，\(R_0,R_1,C_1\) 对 SOC–电流

电阻纵轴为 mΩ，电容为 kF。低温/低 SOC 电阻升高、大电流下 \(R_1\) 下降，都是模型本身的行为。

## 仿真波形

默认读 `Data/nmc100ah_ecm_sim.csv`。先跑仿真再画：

```powershell
python Src/Sim/nmc100ah_ecm_gen.py
python Src/Plot/plot_sim_waveforms.py
python Src/Plot/plot_sim_waveforms.py --csv Data/nmc100ah_ecm_sim.csv --show
```

网格里某一档：

```powershell
python Src/Plot/plot_sim_waveforms.py --csv Data/grid/nmc100ah_ecm_s02_t02_soc050_T+20.csv
```

七路纵排、共用时间轴：

1. 电流（真值 + 测量）
2. 电压（OCV、端电压真值、测量）
3. SOC
4. 极化电压 \(U_p\)
5. \(R_0\)、\(R_1\)
6. \(C_1\)
7. \(\tau_1 = R_1 C_1\)

背景按 `rest` / `discharge` / `charge` 着色。细线是带噪声的测量列。

CSV 须带表头；文件头 `#` 注释会被跳过，与 `nmc100ah_ecm_gen.py` 的输出格式一致。

## 读自己的 CSV

`_common.load_sim_csv(path)` 返回列名到 `numpy` 数组的字典。至少要有波形脚本用到的列：`time_s`、`mode`、`i_true_a`、`i_meas_a`、`u_ocv_v`、`u_t_true_v`、`u_t_meas_v`、`soc_true`、`soc_meas`、`u_p_v`、`r0_ohm`、`r1_ohm`、`c1_f`、`tau1_s`。
