/* 压缩对照：3→16→16→2。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"

static float s_w1[16][3], s_b1[16];
static float s_w2[16][16], s_b2[16];
static float s_w3[2][16], s_b3[2];
static float s_mean[3], s_std[3];

static void fill_mat(float *p, unsigned n, float scale)
{
    unsigned i;
    for (i = 0; i < n; ++i) {
        p[i] = McuEval_RngF(scale);
    }
}

void McuEval_Mlp16_Init(void)
{
    fill_mat(&s_w1[0][0], 16u * 3u, 0.10f);
    fill_mat(s_b1, 16u, 0.02f);
    fill_mat(&s_w2[0][0], 16u * 16u, 0.08f);
    fill_mat(s_b2, 16u, 0.02f);
    fill_mat(&s_w3[0][0], 2u * 16u, 0.08f);
    s_b3[0] = 0.0f;
    s_b3[1] = 0.0f;
    s_mean[0] = 23.0f;
    s_mean[1] = 0.42f;
    s_mean[2] = 21.0f;
    s_std[0] = 59.0f;
    s_std[1] = 0.29f;
    s_std[2] = 19.0f;
}

unsigned McuEval_Mlp16_WeightBytes(void)
{
    return (unsigned)(sizeof(s_w1) + sizeof(s_b1) + sizeof(s_w2) + sizeof(s_b2) + sizeof(s_w3) +
                      sizeof(s_b3) + sizeof(s_mean) + sizeof(s_std));
}

void McuEval_Mlp16_Forward(float i_a, float soc, float t_c, float *r0, float *r1)
{
    float x[3], h1[16], h2[16], z0, z1;
    unsigned i, j;

    x[0] = (i_a - s_mean[0]) / s_std[0];
    x[1] = (soc - s_mean[1]) / s_std[1];
    x[2] = (t_c - s_mean[2]) / s_std[2];
    for (i = 0; i < 16u; ++i) {
        float a = s_b1[i] + s_w1[i][0] * x[0] + s_w1[i][1] * x[1] + s_w1[i][2] * x[2];
        h1[i] = McuEval_Gelu(a);
    }
    for (i = 0; i < 16u; ++i) {
        float a = s_b2[i];
        for (j = 0; j < 16u; ++j) {
            a += s_w2[i][j] * h1[j];
        }
        h2[i] = McuEval_Gelu(a);
    }
    z0 = s_b3[0];
    z1 = s_b3[1];
    for (j = 0; j < 16u; ++j) {
        z0 += s_w3[0][j] * h2[j];
        z1 += s_w3[1][j] * h2[j];
    }
    *r0 = MCU_EVAL_R0_MIN + McuEval_Softplus(z0);
    *r1 = MCU_EVAL_R1_MIN + McuEval_Softplus(z1);
}

void McuEval_Mlp16_Bench(void)
{
    unsigned i, n = 800u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float r0 = 0.0f, r1 = 0.0f, acc = 0.0f;

    McuEval_Mlp16_Init();
    McuEval_ReportHeader("mlp_3x16x16x2");
    McuEval_ReportSize("weights", McuEval_Mlp16_WeightBytes(), "rom");
    McuEval_LogPrintf("n_param=370 note=random_init_count_as_rom\n");
    for (i = 0; i < 4u; ++i) {
        McuEval_Mlp16_Forward(100.0f, 0.5f, 25.0f, &r0, &r1);
    }
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        t0 = McuEval_GetCycles();
        McuEval_Mlp16_Forward(20.0f + (float)(i % 40u), 0.5f, 15.0f, &r0, &r1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += r0 + r1;
    }
    McuEval_ReportCycles("fwd", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.9f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_Mlp16_Bench();
    return 0;
}
#endif
