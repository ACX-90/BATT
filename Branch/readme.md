Branch文件夹供Git分支开发使用，若处于分支状态，未经同意，只能在这个文件夹增删文件
若允许合并主线，再把变更移到对应文件夹替换对应文件

`eval_pybamm_rc/`：从已有 `Data/grid_pybamm` 序列估 R0/R1/R2/τ1/τ2。
结论 `eval_pybamm_rc/eval.md`；NLS 专项 `eval_pybamm_rc/eval_nls.md`。
复现 `python Branch/eval_pybamm_rc/identify_rc.py` 与 `python Branch/eval_pybamm_rc/eval_nls.py`。