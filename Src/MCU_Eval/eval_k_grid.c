/* (SOC,T) 上 5×4 k 网格，双线性插值。复制：mcu_eval_common.* + 本文件。 */
#include "mcu_eval_algos.h"

static const float k_soc_node[MCU_KGRID_NS] = {0.10f, 0.30f, 0.50f, 0.70f, 0.90f};
static const float k_t_node[MCU_KGRID_NT] = {-10.0f, 10.0f, 30.0f, 50.0f};

void McuEval_KGrid_Init(McuKGrid *g)
{
    unsigned i, j;
    for (i = 0; i < MCU_KGRID_NS; ++i) {
        for (j = 0; j < MCU_KGRID_NT; ++j) {
            g->k0[i][j] = 1.0f;
            g->k1[i][j] = 1.0f;
        }
    }
}

void McuEval_KGrid_InterpW(float soc, float t_c, int *is, int *it, float *ws, float *wt)
{
    int i, j;
    soc = McuEval_Clampf(soc, k_soc_node[0], k_soc_node[MCU_KGRID_NS - 1]);
    t_c = McuEval_Clampf(t_c, k_t_node[0], k_t_node[MCU_KGRID_NT - 1]);
    i = 0;
    while (i < MCU_KGRID_NS - 2 && soc > k_soc_node[i + 1]) {
        ++i;
    }
    j = 0;
    while (j < MCU_KGRID_NT - 2 && t_c > k_t_node[j + 1]) {
        ++j;
    }
    *is = i;
    *it = j;
    *ws = (soc - k_soc_node[i]) / (k_soc_node[i + 1] - k_soc_node[i]);
    *wt = (t_c - k_t_node[j]) / (k_t_node[j + 1] - k_t_node[j]);
}

void McuEval_KGrid_Apply(const McuKGrid *g, float soc, float t_c, float r0, float r1,
                         float *o0, float *o1)
{
    int is, it;
    float ws, wt, w00, w10, w01, w11, k0, k1;
    McuEval_KGrid_InterpW(soc, t_c, &is, &it, &ws, &wt);
    w00 = (1.0f - ws) * (1.0f - wt);
    w10 = ws * (1.0f - wt);
    w01 = (1.0f - ws) * wt;
    w11 = ws * wt;
    k0 = w00 * g->k0[is][it] + w10 * g->k0[is + 1][it] + w01 * g->k0[is][it + 1] +
         w11 * g->k0[is + 1][it + 1];
    k1 = w00 * g->k1[is][it] + w10 * g->k1[is + 1][it] + w01 * g->k1[is][it + 1] +
         w11 * g->k1[is + 1][it + 1];
    *o0 = k0 * r0;
    *o1 = k1 * r1;
}

void McuEval_KGrid_Bench(void)
{
    McuKGrid g;
    unsigned i, n = 2000u;
    uint32_t cmin = 0xFFFFFFFFu;
    unsigned long long csum = 0;
    float o0 = 0.0f, o1 = 0.0f, acc = 0.0f;

    McuEval_KGrid_Init(&g);
    g.k0[0][0] = 1.20f;
    g.k1[4][3] = 1.10f;
    McuEval_ReportHeader("k_grid_5x4");
    McuEval_ReportSize("per_cell", (unsigned)sizeof(McuKGrid), "ram");
    McuEval_LogPrintf("n_node=%u n_k=%u\n", (unsigned)(MCU_KGRID_NS * MCU_KGRID_NT),
                      (unsigned)(2u * MCU_KGRID_NS * MCU_KGRID_NT));
    for (i = 0; i < n; ++i) {
        uint32_t t0, dt;
        float s = 0.15f + 0.01f * (float)(i % 70u);
        float t = -5.0f + 0.5f * (float)(i % 80u);
        t0 = McuEval_GetCycles();
        McuEval_KGrid_Apply(&g, s, t, 8.0e-4f, 6.5e-4f, &o0, &o1);
        dt = McuEval_GetCycles() - t0;
        if (dt < cmin) {
            cmin = dt;
        }
        csum += dt;
        acc += o0 + o1;
    }
    McuEval_ReportCycles("interp_apply", n, 1u, cmin, csum);
    McuEval_LogPrintf("sink=%.9f pack180_B=%u\n", (double)acc,
                      (unsigned)sizeof(McuKGrid) * MCU_EVAL_N_CELL);
}

#ifdef MCU_EVAL_STANDALONE
int main(void)
{
    McuEval_Init();
    McuEval_KGrid_Bench();
    return 0;
}
#endif
