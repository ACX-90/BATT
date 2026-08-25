/* 把 64×64 烤成均匀轴查找表：2D 双线性、3D 三线性。
 * 权重随机填，只测负荷。复制：mcu_eval_common.* + 本文件。
 *
 * 轴与 bake_lut.py 默认盒一致：I∈[-200,200]、SOC∈[0.05,0.95]、T∈[-10,50]。
 * 3D 默认 9×11×9（推荐档）；2D 默认 21×13（对照，忽略电流维）。
 */
#include "mcu_eval_algos.h"

#define MCU_LUT2_NS 21
#define MCU_LUT2_NT 13
#define MCU_LUT2_S0 0.05f
#define MCU_LUT2_DS (0.90f / 20.0f)
#define MCU_LUT2_T0 (-10.0f)
#define MCU_LUT2_DT (60.0f / 12.0f)

#define MCU_LUT3_NI 9
#define MCU_LUT3_NS 11
#define MCU_LUT3_NT 9
#define MCU_LUT3_I0 (-200.0f)
#define MCU_LUT3_DI (400.0f / 8.0f)
#define MCU_LUT3_S0 0.05f
#define MCU_LUT3_DS (0.90f / 10.0f)
#define MCU_LUT3_T0 (-10.0f)
#define MCU_LUT3_DT (60.0f / 8.0f)

static float s_tab2[MCU_LUT2_NS][MCU_LUT2_NT][2];
static float s_tab3[MCU_LUT3_NI][MCU_LUT3_NS][MCU_LUT3_NT][2];

static void loc1(float x, float x0, float dx, unsigned n, unsigned *i0, float *f)
{
    float t = (x - x0) / dx;
    if (t < 0.0f) {
        t = 0.0f;
    }
    if (t > (float)(n - 1) - 1.0e-6f) {
        t = (float)(n - 1) - 1.0e-6f;
    }
    *i0 = (unsigned)t;
    *f = t - (float)(*i0);
}

void McuEval_Lut2_Init(void)
{
    unsigned i, j, k;
    for (i = 0; i < MCU_LUT2_NS; ++i) {
        for (j = 0; j < MCU_LUT2_NT; ++j) {
            for (k = 0; k < 2u; ++k) {
                s_tab2[i][j][k] = 6.0e-4f + 2.0e-4f * McuEval_RngF(1.0f);
            }
        }
    }
}

void McuEval_Lut3_Init(void)
{
    unsigned i, j, k, c;
    for (i = 0; i < MCU_LUT3_NI; ++i) {
        for (j = 0; j < MCU_LUT3_NS; ++j) {
            for (k = 0; k < MCU_LUT3_NT; ++k) {
                for (c = 0; c < 2u; ++c) {
                    s_tab3[i][j][k][c] = 6.0e-4f + 2.0e-4f * McuEval_RngF(1.0f);
                }
            }
        }
    }
}

unsigned McuEval_Lut2_WeightBytes(void)
{
    return (unsigned)sizeof(s_tab2);
}

unsigned McuEval_Lut3_WeightBytes(void)
{
    return (unsigned)sizeof(s_tab3);
}

void McuEval_Lut2_Forward(float soc, float t_c, float *r0, float *r1)
{
    unsigned is, it;
    float ws, wt, w00, w10, w01, w11;
    loc1(soc, MCU_LUT2_S0, MCU_LUT2_DS, MCU_LUT2_NS, &is, &ws);
    loc1(t_c, MCU_LUT2_T0, MCU_LUT2_DT, MCU_LUT2_NT, &it, &wt);
    w00 = (1.0f - ws) * (1.0f - wt);
    w10 = ws * (1.0f - wt);
    w01 = (1.0f - ws) * wt;
    w11 = ws * wt;
    *r0 = w00 * s_tab2[is][it][0] + w10 * s_tab2[is + 1][it][0] + w01 * s_tab2[is][it + 1][0] +
          w11 * s_tab2[is + 1][it + 1][0];
    *r1 = w00 * s_tab2[is][it][1] + w10 * s_tab2[is + 1][it][1] + w01 * s_tab2[is][it + 1][1] +
          w11 * s_tab2[is + 1][it + 1][1];
}

void McuEval_Lut3_Forward(float i_a, float soc, float t_c, float *r0, float *r1)
{
    unsigned ii, is, it, c;
    float wi, ws, wt;
    float out[2];
    loc1(i_a, MCU_LUT3_I0, MCU_LUT3_DI, MCU_LUT3_NI, &ii, &wi);
    loc1(soc, MCU_LUT3_S0, MCU_LUT3_DS, MCU_LUT3_NS, &is, &ws);
    loc1(t_c, MCU_LUT3_T0, MCU_LUT3_DT, MCU_LUT3_NT, &it, &wt);
    out[0] = 0.0f;
    out[1] = 0.0f;
    {
        float w[2][2][2];
        unsigned di, ds, dt;
        w[0][0][0] = (1.0f - wi) * (1.0f - ws) * (1.0f - wt);
        w[1][0][0] = wi * (1.0f - ws) * (1.0f - wt);
        w[0][1][0] = (1.0f - wi) * ws * (1.0f - wt);
        w[1][1][0] = wi * ws * (1.0f - wt);
        w[0][0][1] = (1.0f - wi) * (1.0f - ws) * wt;
        w[1][0][1] = wi * (1.0f - ws) * wt;
        w[0][1][1] = (1.0f - wi) * ws * wt;
        w[1][1][1] = wi * ws * wt;
        for (di = 0; di < 2u; ++di) {
            for (ds = 0; ds < 2u; ++ds) {
                for (dt = 0; dt < 2u; ++dt) {
                    float ww = w[di][ds][dt];
                    for (c = 0; c < 2u; ++c) {
                        out[c] += ww * s_tab3[ii + di][is + ds][it + dt][c];
                    }
                }
            }
        }
    }
    *r0 = out[0];
    *r1 = out[1];
}

void McuEval_Lut2_Bench(void)
{
    unsigned i, n = 4000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float r0 = 0.0f, r1 = 0.0f, acc = 0.0f;

    McuEval_Lut2_Init();
    McuEval_ReportHeader("lut2d_21x13");
    McuEval_ReportSize("table", McuEval_Lut2_WeightBytes(), "rom");
    McuEval_LogPrintf("n_node=%u n_ch=2 note=bilinear_ignore_I\n",
                      (unsigned)(MCU_LUT2_NS * MCU_LUT2_NT));
    for (i = 0; i < 4u; ++i) {
        McuEval_Lut2_Forward(0.50f, 25.0f, &r0, &r1);
    }
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        t0 = McuEval_GetCycles();
        McuEval_Lut2_Forward(0.15f + 0.01f * (float)(i % 70u), -5.0f + 0.5f * (float)(i % 80u), &r0,
                             &r1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += r0 + r1;
    }
    McuEval_ReportCycles("interp2", n, 1u, cmin, csum);
    McuEval_ReportCycles("interp2_if_180seq", n, MCU_EVAL_N_CELL, cmin, csum);
    McuEval_LogPrintf("sink=%.9f\n", (double)acc);
}

void McuEval_Lut3_Bench(void)
{
    unsigned i, n = 4000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float r0 = 0.0f, r1 = 0.0f, acc = 0.0f;

    McuEval_Lut3_Init();
    McuEval_ReportHeader("lut3d_9x11x9");
    McuEval_ReportSize("table", McuEval_Lut3_WeightBytes(), "rom");
    McuEval_LogPrintf("n_node=%u n_ch=2 note=trilinear_I_SOC_T\n",
                      (unsigned)(MCU_LUT3_NI * MCU_LUT3_NS * MCU_LUT3_NT));
    for (i = 0; i < 4u; ++i) {
        McuEval_Lut3_Forward(100.0f, 0.50f, 25.0f, &r0, &r1);
    }
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        t0 = McuEval_GetCycles();
        McuEval_Lut3_Forward(-180.0f + (float)(i % 90u) * 4.0f, 0.15f + 0.01f * (float)(i % 70u),
                             -5.0f + 0.5f * (float)(i % 80u), &r0, &r1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += r0 + r1;
    }
    McuEval_ReportCycles("interp3", n, 1u, cmin, csum);
    McuEval_ReportCycles("interp3_if_180seq", n, MCU_EVAL_N_CELL, cmin, csum);
    McuEval_LogPrintf("sink=%.9f\n", (double)acc);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_Lut2_Bench();
    McuEval_Lut3_Bench();
    return 0;
}
#endif
