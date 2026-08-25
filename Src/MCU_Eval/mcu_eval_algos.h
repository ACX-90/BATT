#ifndef MCU_EVAL_ALGOS_H
#define MCU_EVAL_ALGOS_H

#include "mcu_eval_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float s;
    float u_p;
    float d_r0;
    float P[2][2];
    float i_a;
    float s_pred;
    int use_dr0;
} McuEkf;

#define MCU_KGRID_NS 5
#define MCU_KGRID_NT 4

typedef struct {
    float k0[MCU_KGRID_NS][MCU_KGRID_NT];
    float k1[MCU_KGRID_NS][MCU_KGRID_NT];
} McuKGrid;

void McuEval_Ekf_Reset(McuEkf *e, float s0, int use_dr0);
void McuEval_Ekf_Step(McuEkf *e, float i_a, float t_c, float u_meas, float r0, float r1,
                      float c1);
void McuEval_Ekf_Bench(void);
void McuEval_EkfDr0_Bench(void);

void McuEval_Mlp64_Init(void);
void McuEval_Mlp64_Forward(float i_a, float soc, float t_c, float *r0, float *r1);
unsigned McuEval_Mlp64_WeightBytes(void);
void McuEval_Mlp64_Bench(void);

void McuEval_Mlp16_Init(void);
void McuEval_Mlp16_Forward(float i_a, float soc, float t_c, float *r0, float *r1);
unsigned McuEval_Mlp16_WeightBytes(void);
void McuEval_Mlp16_Bench(void);

void McuEval_Head382_Init(void);
void McuEval_Head382_Forward(float i_a, float soc, float t_c, float *dr0, float *dr1);
unsigned McuEval_Head382_WeightBytes(void);
void McuEval_Head382_Bench(void);

void McuEval_KGlobal_Init(float *k0, float *k1);
void McuEval_KGlobal_Apply(float k0, float k1, float r0, float r1, float *o0, float *o1);
void McuEval_KGlobal_Bench(void);

void McuEval_KGrid_Init(McuKGrid *g);
void McuEval_KGrid_Apply(const McuKGrid *g, float soc, float t_c, float r0, float r1,
                         float *o0, float *o1);
void McuEval_KGrid_InterpW(float soc, float t_c, int *is, int *it, float *ws, float *wt);
void McuEval_KGrid_Bench(void);

void McuEval_Lut2_Init(void);
void McuEval_Lut2_Forward(float soc, float t_c, float *r0, float *r1);
unsigned McuEval_Lut2_WeightBytes(void);
void McuEval_Lut2_Bench(void);

void McuEval_Lut3_Init(void);
void McuEval_Lut3_Forward(float i_a, float soc, float t_c, float *r0, float *r1);
unsigned McuEval_Lut3_WeightBytes(void);
void McuEval_Lut3_Bench(void);

void McuEval_Ecm_Bench(void);
void McuEval_PulseR0_Bench(void);
void McuEval_IncrKGrid_Bench(void);
void McuEval_SoxCell_Bench(void);
void McuEval_SoxPack_Bench(void);

#ifdef __cplusplus
}
#endif
#endif
