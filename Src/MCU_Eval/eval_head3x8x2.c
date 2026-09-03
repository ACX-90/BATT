/* 英飞凌 ifx_demo 残差头 3→8→2。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"
#include <math.h>

static float s_w1[8][3], s_b1[8];
static float s_w2[2][8], s_b2[2];

void McuEval_Head382_Init(void)
{
    unsigned i;
    for (i = 0; i < 8u * 3u; ++i) {
        (&s_w1[0][0])[i] = McuEval_RngF(0.15f);
    }
    for (i = 0; i < 8u; ++i) {
        s_b1[i] = McuEval_RngF(0.05f);
    }
    for (i = 0; i < 2u * 8u; ++i) {
        (&s_w2[0][0])[i] = McuEval_RngF(0.10f);
    }
    s_b2[0] = 0.0f;
    s_b2[1] = 0.0f;
}

unsigned McuEval_Head382_WeightBytes(void)
{
    return (unsigned)(sizeof(s_w1) + sizeof(s_b1) + sizeof(s_w2) + sizeof(s_b2));
}

void McuEval_Head382_Forward(float i_a, float soc, float t_c, float *dr0, float *dr1)
{
    float x[3] = {i_a / 100.0f, soc, (t_c + 20.0f) / 70.0f};
    float h[8];
    unsigned i, j;
    float z0, z1;

    for (i = 0; i < 8u; ++i) {
        float a = s_b1[i] + s_w1[i][0] * x[0] + s_w1[i][1] * x[1] + s_w1[i][2] * x[2];
        h[i] = tanhf(a);
    }
    z0 = s_b2[0];
    z1 = s_b2[1];
    for (j = 0; j < 8u; ++j) {
        z0 += s_w2[0][j] * h[j];
        z1 += s_w2[1][j] * h[j];
    }
    *dr0 = z0;
    *dr1 = z1;
}

void McuEval_Head382_Bench(void)
{
    unsigned i, n = 2000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float d0 = 0.0f, d1 = 0.0f, acc = 0.0f;

    McuEval_Head382_Init();
    McuEval_ReportHeader("head_3x8x2");
    McuEval_ReportSize("weights", McuEval_Head382_WeightBytes(), "rom");
    McuEval_LogPrintf("n_param=50 per_cell_head=18floats\n");
    for (i = 0; i < 8u; ++i) {
        McuEval_Head382_Forward(100.0f, 0.5f, 25.0f, &d0, &d1);
    }
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        t0 = McuEval_GetCycles();
        McuEval_Head382_Forward(50.0f, 0.4f, 10.0f, &d0, &d1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += d0 + d1;
    }
    McuEval_ReportCycles("fwd", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.9f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_Head382_Bench();
    return 0;
}
#endif
