# Phase-3 · 1RC + AI δU（PyBaMM 主列）

方案见 `Doc/07-c`，数字见 `Doc/07-d`。

## 跑

仓库根目录（需 `torch` / `scipy`；生成网格另需 `pybamm`）：

```bash
python Src/Sim/nmc100ah_gen_grid.py --pybamm --n-soc 5 --n-temp 5 --out-dir Data/grid_pybamm --no-noise
python Branch/phase3_1rc_ai/run_3b.py --epochs-mlp 12 --epochs-gru 25 --clip-mv 8
```

- **不读** `Data/ai_mlp`。底板写在 `out/mlp_pybamm`，LUT 在 `out/lut_pybamm`。
- 摘要：`out/metrics_3b.json`；明细：`out/summary_3b.json`。

## 模块

| 文件 | 作用 |
|------|------|
| `run_3b.py` | 3B1–3B4 编排 |
| `residual_gru.py` | 因果 GRU \(d=4\) → 标量 δU |
| `rc_sim.py` | 1RC/2RC 开环、分段 RMSE、LUT 插值 |
