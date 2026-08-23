/* 180 芯顺序一拍：共享 MLP64 + 每芯 k 网格 + 每芯 EKF。
 * 复制：mcu_eval_common.* + eval_mlp64.c + eval_ekf.c + eval_k_grid.c + 本文件。
 * 内存按真包铺开，用来对照 RAM；周期是 180 次串行前向。
 */
#include "mcu_eval_algos.h"

static McuEkf s_ekf[MCU_EVAL_N_CELL];
static McuKGrid s_grid[MCU_EVAL_N_CELL];

void McuEval_SoxPack_Bench(void)
{
    unsigned c, n = 20u, it;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float acc = 0.0f;

    McuEval_Mlp64_Init();
    for (c = 0; c < MCU_EVAL_N_CELL; ++c) {
        McuEval_KGrid_Init(&s_grid[c]);
        McuEval_Ekf_Reset(&s_ekf[c], 0.65f + 0.0002f * (float)c, 0);
        s_grid[c].k0[2][2] = 1.0f + 0.001f * (float)(c % 17u);
    }
    McuEval_ReportHeader("sox_pack180_mlp64_kgrid_ekf");
    McuEval_ReportSize("mlp_weights_shared", McuEval_Mlp64_WeightBytes(), "rom");
    McuEval_ReportSize("kgrid_all", (unsigned)sizeof(s_grid), "ram");
    McuEval_ReportSize("ekf_all", (unsigned)sizeof(s_ekf), "ram");
    McuEval_ReportSize("logbuf_eval_only", MCU_EVAL_LOG_BYTES, "ram_subtract");

    for (it = 0; it < n; ++it) {
        uint32_t t0, dt;
        float ii = 70.0f;
        t0 = McuEval_GetCycles();
        for (c = 0; c < MCU_EVAL_N_CELL; ++c) {
            float r0, r1, o0, o1;
            float t_c = 20.0f + 0.02f * (float)(c % 30u);
            McuEval_Mlp64_Forward(ii, s_ekf[c].s, t_c, &r0, &r1);
            McuEval_KGrid_Apply(&s_grid[c], s_ekf[c].s, t_c, r0, r1, &o0, &o1);
            McuEval_Ekf_Step(&s_ekf[c], ii, t_c, 3.67f, o0, o1, MCU_EVAL_C1_STAR);
        }
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += s_ekf[0].s;
    }
    McuEval_ReportCycles("tick_pack180", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_SoxPack_Bench();
    return 0;
}
#endif
