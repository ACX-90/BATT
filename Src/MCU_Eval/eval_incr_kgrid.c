/* 滑窗累积 k 网格梯度再一步 SGD（停车/降频，不进 0.1 s 环）。
 * 复制：mcu_eval_common.* + eval_k_grid.c + 本文件。
 */
#include "mcu_eval_algos.h"

#ifndef MCU_INCR_WIN
#define MCU_INCR_WIN 100
#endif

void McuEval_IncrKGrid_Bench(void)
{
    McuKGrid g;
    float g0[MCU_KGRID_NS][MCU_KGRID_NT];
    float g1[MCU_KGRID_NS][MCU_KGRID_NT];
    unsigned i, n = 40u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float acc = 0.0f;
    const float lr = 0.05f;

    McuEval_KGrid_Init(&g);
    McuEval_ReportHeader("incr_kgrid_win100");
    McuEval_ReportSize("scratch_grad", (unsigned)(sizeof(g0) + sizeof(g1)), "ram");
    McuEval_LogPrintf("win=%u note=not_in_10Hz_loop\n", (unsigned)MCU_INCR_WIN);

    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        unsigned k;
        int r, c;
        t0 = McuEval_GetCycles();
        for (r = 0; r < MCU_KGRID_NS; ++r) {
            for (c = 0; c < MCU_KGRID_NT; ++c) {
                g0[r][c] = 0.0f;
                g1[r][c] = 0.0f;
            }
        }
        for (k = 0; k < (unsigned)MCU_INCR_WIN; ++k) {
            int is, it;
            float ws, wt, w00, w10, w01, w11;
            float r0 = 8.0e-4f, r1 = 6.5e-4f, o0, o1, e, i_a = 80.0f;
            float soc = 0.4f + 0.001f * (float)k;
            float t_c = 20.0f;
            McuEval_KGrid_Apply(&g, soc, t_c, r0, r1, &o0, &o1);
            e = 0.012f - i_a * (o0 - r0); /* 假开环残差，只为乘加计数 */
            McuEval_KGrid_InterpW(soc, t_c, &is, &it, &ws, &wt);
            w00 = (1.0f - ws) * (1.0f - wt);
            w10 = ws * (1.0f - wt);
            w01 = (1.0f - ws) * wt;
            w11 = ws * wt;
            {
                float d = -e * i_a * r0;
                g0[is][it] += d * w00;
                g0[is + 1][it] += d * w10;
                g0[is][it + 1] += d * w01;
                g0[is + 1][it + 1] += d * w11;
            }
        }
        for (r = 0; r < MCU_KGRID_NS; ++r) {
            for (c = 0; c < MCU_KGRID_NT; ++c) {
                g.k0[r][c] = McuEval_Clampf(g.k0[r][c] - lr * g0[r][c], 0.5f, 2.0f);
            }
        }
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += g.k0[2][1];
    }
    McuEval_ReportCycles("window_sgd", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_IncrKGrid_Bench();
    return 0;
}
#endif
