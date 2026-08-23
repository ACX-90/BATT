/* 全局 k0,k1。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"

void McuEval_KGlobal_Init(float *k0, float *k1)
{
    *k0 = 1.0f;
    *k1 = 1.0f;
}

void McuEval_KGlobal_Apply(float k0, float k1, float r0, float r1, float *o0, float *o1)
{
    *o0 = k0 * r0;
    *o1 = k1 * r1;
}

void McuEval_KGlobal_Bench(void)
{
    unsigned i, n = 3000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float k0, k1, o0 = 0.0f, o1 = 0.0f, acc = 0.0f;

    McuEval_KGlobal_Init(&k0, &k1);
    k0 = 1.15f;
    k1 = 1.15f;
    McuEval_ReportHeader("k_global");
    McuEval_ReportSize("per_cell", 8u, "ram");
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        t0 = McuEval_GetCycles();
        McuEval_KGlobal_Apply(k0, k1, 8.0e-4f, 6.5e-4f, &o0, &o1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += o0 + o1;
    }
    McuEval_ReportCycles("apply", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.9f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_KGlobal_Bench();
    return 0;
}
#endif
