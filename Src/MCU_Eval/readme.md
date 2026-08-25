# Src/MCU_Eval — 拷到英飞凌工程里测周期 / 负载

配套 [`Doc/05-c-TC4D7资源与负载评估.md`](../../Doc/05-c-TC4D7资源与负载评估.md)。

每个算法一份 `eval_*.c`。权重用随机数填，不追求数值对。日志写进 `g_mcu_log[]`（默认 64 KB），你从串口整块发出去即可。

## 拷进 AURIX 工程

1. 始终加入 `mcu_eval_common.c`、`mcu_eval_common.h`、`mcu_eval_algos.h`
2. 按要测的算法加入对应 `eval_*.c`（见下表）。**不要**同时定义多个 `MCU_EVAL_STANDALONE`（会有多个 `main`）
3. 在 `Cpu0_Main.c` 时钟稳定后：

```c
McuEval_Init();
McuEval_Mlp64_Bench();   /* 或其它 *_Bench */
/* 把 McuEval_LogBuf() 的 McuEval_LogLen() 字节从 ASCLIN 发出 */
```

一次跑完全部：加入所有 `eval_*.c` **以及** `eval_all.c`，不要给单个算法加 `MCU_EVAL_STANDALONE`。

| 测这个 | 还要一起加的算法 .c |
|--------|---------------------|
| `eval_ecm.c` / `eval_ekf.c` / `eval_mlp64.c` / `eval_mlp16.c` / `eval_lut.c`（`lut2d_21x13` / `lut3d_9x11x9`） / `eval_head3x8x2.c` / `eval_k_global.c` / `eval_k_grid.c` / `eval_pulse_r0.c` | 无 |
| `eval_incr_kgrid.c` | `eval_k_grid.c` |
| `eval_sox_cell.c` / `eval_sox_pack.c` | `eval_mlp64.c` `eval_ekf.c` `eval_k_grid.c` |

`McuEval_Init` 会打开 CPU_CCNT（CCTRL=`0xFC00`，CCNT=`0xFC04`）。AURIX GCC 必须用立即数 `mfcr %0, %1` + `"i"(0xFC04)`，不能写 `$FC04`（会被当成链接符号）。若和 iLLD 冲突，改 `McuEval_GetCycles` 即可。

## 主机冒烟

仓库根目录：

```text
Script\mcu_eval_host.bat
```

需要 GCC。芯片上的周期数以 TC4D7 为准，主机 rdtsc 只能看代码能跑通。
