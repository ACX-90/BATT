/* 二维 EKF (s, Up)，可选 δR0。对齐 Src/AI/KF/ekf.py 的公式，OCV 用评测用小表。
 * 复制：mcu_eval_common.* + 本文件。
 */
#include "mcu_eval_algos.h"
#include <math.h>

void McuEval_Ekf_Reset(McuEkf *e, float s0, int use_dr0)
{
    e->s = McuEval_Clampf(s0, 0.0f, 1.0f);
    e->u_p = 0.0f;
    e->d_r0 = 0.0f;
    e->P[0][0] = 2.5e-3f;
    e->P[0][1] = 0.0f;
    e->P[1][0] = 0.0f;
    e->P[1][1] = 2.5e-3f;
    e->i_a = 0.0f;
    e->s_pred = e->s;
    e->use_dr0 = use_dr0;
}

void McuEval_Ekf_Step(McuEkf *e, float i_a, float t_c, float u_meas, float r0, float r1,
                      float c1)
{
    const float q_s = 1.0e-8f;
    const float q_up = 1.0e-6f;
    const float rv0 = 0.5e-3f * 0.5e-3f;
    float s_pred, r0u, u_ocv, slope, u_p_pred, u_t_pri, alpha, e_pri;
    float F11, Pp00, Pp01, Pp10, Pp11;
    float rv, H0, H1, s_inn, K0, K1;
    float IKH00, IKH01, IKH10, IKH11;

    e->i_a = i_a;
    s_pred = McuEval_Clampf(e->s - i_a * MCU_EVAL_DT / MCU_EVAL_Q_COULOMB, 0.0f, 1.0f);
    e->s_pred = s_pred;

    r0u = r0 + (e->use_dr0 ? e->d_r0 : 0.0f);
    u_ocv = McuEval_Ocv(s_pred, t_c);
    slope = McuEval_DocvDs(s_pred, t_c);
    McuEval_Ecm1(i_a, e->u_p, r0u, r1, c1, u_ocv, &u_p_pred, &u_t_pri, &alpha);
    e_pri = u_meas - u_t_pri;

    F11 = alpha;
    Pp00 = e->P[0][0] + q_s;
    Pp01 = e->P[0][1] * F11;
    Pp10 = F11 * e->P[1][0];
    Pp11 = F11 * e->P[1][1] * F11 + q_up;

    rv = rv0;
    {
        float sc = 0.20f / (fabsf(slope) + 1.0e-6f);
        sc = sc * sc;
        if (sc < 1.0f) {
            sc = 1.0f;
        }
        if (sc > 25.0f) {
            sc = 25.0f;
        }
        rv *= sc;
    }
    H0 = slope;
    H1 = -1.0f;
    s_inn = H0 * (H0 * Pp00 + H1 * Pp10) + H1 * (H0 * Pp01 + H1 * Pp11) + rv;
    if (s_inn < 1.0e-18f) {
        s_inn = 1.0e-18f;
    }
    K0 = (Pp00 * H0 + Pp01 * H1) / s_inn;
    K1 = (Pp10 * H0 + Pp11 * H1) / s_inn;
    K0 = McuEval_Clampf(K0, -2.0f, 2.0f);

    e->s = McuEval_Clampf(s_pred + K0 * e_pri, 0.0f, 1.0f);
    e->u_p = u_p_pred + K1 * e_pri;

    IKH00 = 1.0f - K0 * H0;
    IKH01 = -K0 * H1;
    IKH10 = -K1 * H0;
    IKH11 = 1.0f - K1 * H1;
    e->P[0][0] = IKH00 * Pp00 + IKH01 * Pp10 + K0 * rv * K0;
    e->P[0][1] = IKH00 * Pp01 + IKH01 * Pp11 + K0 * rv * K1;
    e->P[1][0] = IKH10 * Pp00 + IKH11 * Pp10 + K1 * rv * K0;
    e->P[1][1] = IKH10 * Pp01 + IKH11 * Pp11 + K1 * rv * K1;
}

static void bench_ekf(const char *name, int use_dr0)
{
    McuEkf e;
    unsigned i, n = 1500u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float acc = 0.0f;

    McuEval_Ekf_Reset(&e, 0.70f, use_dr0);
    McuEval_ReportHeader(name);
    McuEval_ReportSize("state", (unsigned)sizeof(McuEkf), "ram");
    for (i = 0; i < 8u; ++i) {
        McuEval_Ekf_Step(&e, 50.0f, 25.0f, 3.70f, 8.0e-4f, 6.5e-4f, MCU_EVAL_C1_STAR);
    }
    McuEval_Ekf_Reset(&e, 0.70f, use_dr0);
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        float ii = 40.0f + (float)(i % 17u);
        t0 = McuEval_GetCycles();
        McuEval_Ekf_Step(&e, ii, 25.0f, 3.68f, 8.0e-4f, 6.5e-4f, MCU_EVAL_C1_STAR);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += e.s;
    }
    McuEval_ReportCycles("step", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.6f\n", (double)acc);
}

void McuEval_Ekf_Bench(void) { bench_ekf("ekf_s_up", 0); }
void McuEval_EkfDr0_Bench(void) { bench_ekf("ekf_s_up_dr0", 1); }

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_Ekf_Bench();
    McuEval_EkfDr0_Bench();
    return 0;
}
#endif
