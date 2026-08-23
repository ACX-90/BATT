/* 边沿 ΔU/ΔI 估 k0，无 ECM 反传。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"

static float pulse_k0(float du, float di, float r0_fleet)
{
    float rhat;
    if (di > -1.0f && di < 1.0f) {
        return 1.0f;
    }
    rhat = -du / di;
    if (rhat < 1.0e-5f) {
        rhat = 1.0e-5f;
    }
    return rhat / (r0_fleet + 1.0e-12f);
}

void McuEval_PulseR0_Bench(void)
{
    unsigned i, n = 3000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float acc = 0.0f;

    McuEval_ReportHeader("pulse_dU_dI_k0");
    McuEval_ReportSize("per_cell", 4u, "ram");
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        float k;
        t0 = McuEval_GetCycles();
        k = pulse_k0(-0.08f, 100.0f, 8.0e-4f);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += k;
    }
    McuEval_ReportCycles("step", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_PulseR0_Bench();
    return 0;
}
#endif
