/* 单芯一拍：MLP64 + k 网格 + ECM/EKF。
 * 复制：mcu_eval_common.* + eval_mlp64.c + eval_ekf.c + eval_k_grid.c + 本文件。
 */
#include "mcu_eval_algos.h"

void McuEval_SoxCell_Bench(void)
{
    McuEkf ekf;
    McuKGrid grid;
    unsigned i, n = 200u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float acc = 0.0f;

    McuEval_Mlp64_Init();
    McuEval_KGrid_Init(&grid);
    McuEval_Ekf_Reset(&ekf, 0.70f, 0);
    McuEval_ReportHeader("sox_cell_mlp64_kgrid_ekf");
    McuEval_ReportSize("mlp_weights", McuEval_Mlp64_WeightBytes(), "rom");
    McuEval_ReportSize("kgrid", (unsigned)sizeof(grid), "ram");
    McuEval_ReportSize("ekf", (unsigned)sizeof(ekf), "ram");

    for (i = 0; i < 4u; ++i) {
        float r0, r1, o0, o1;
        McuEval_Mlp64_Forward(60.0f, ekf.s, 25.0f, &r0, &r1);
        McuEval_KGrid_Apply(&grid, ekf.s, 25.0f, r0, r1, &o0, &o1);
        McuEval_Ekf_Step(&ekf, 60.0f, 25.0f, 3.69f, o0, o1, MCU_EVAL_C1_STAR);
    }
    McuEval_Ekf_Reset(&ekf, 0.70f, 0);
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        float r0, r1, o0, o1;
        float ii = 50.0f + (float)(i % 20u);
        t0 = McuEval_GetCycles();
        McuEval_Mlp64_Forward(ii, ekf.s, 25.0f, &r0, &r1);
        McuEval_KGrid_Apply(&grid, ekf.s, 25.0f, r0, r1, &o0, &o1);
        McuEval_Ekf_Step(&ekf, ii, 25.0f, 3.68f, o0, o1, MCU_EVAL_C1_STAR);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += ekf.s;
    }
    McuEval_ReportCycles("tick_1cell", n, 1u, cmin, csum);
    McuEval_ReportCycles("tick_if_180seq", n, MCU_EVAL_N_CELL, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_SoxCell_Bench();
    return 0;
}
#endif
