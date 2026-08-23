/* 一阶 Thevenin 单步。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"

void McuEval_Ecm_Bench(void)
{
    unsigned i;
    unsigned n = 2000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float u_p = 0.0f, u_t = 0.0f, a = 0.0f, acc = 0.0f;

    McuEval_ReportHeader("ecm1");
    McuEval_ReportSize("payload", 0u, "rom");
    for (i = 0; i < 8u; ++i) {
        McuEval_Ecm1(100.0f, u_p, 8.0e-4f, 6.5e-4f, MCU_EVAL_C1_STAR, 3.7f, &u_p, &u_t, &a);
    }
    u_p = 0.0f;
    for (i = 0; i < n; ++i) {
        uint32_t t0 = McuEval_GetCycles();
        McuEval_Ecm1(80.0f + (float)(i & 7u), u_p, 8.0e-4f, 6.5e-4f, MCU_EVAL_C1_STAR, 3.7f,
                     &u_p, &u_t, &a);
        {
            uint32_t dt = McuEval_GetCycles() - t0;
            if (dt < cmin) {
                cmin = dt;
            }
            csum += dt;
        }
        acc += u_t;
    }
    McuEval_ReportCycles("step", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_Ecm_Bench();
    return 0;
}
#endif
