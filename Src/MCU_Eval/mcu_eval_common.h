/* TC4D7 / 主机通用：周期计数、日志缓冲、初等函数。
 * 算法文件只依赖这一份头；实现在 mcu_eval_common.c。
 *
 * 英飞凌工程：把本文件和 mcu_eval_common.c 与要测的 eval_*.c 一起加入。
 * 时钟起来之后调用 McuEval_Init()，再调对应 Bench，最后把
 * McuEval_LogBuf()[0 .. McuEval_LogLen()) 从串口发出去。
 */
#ifndef MCU_EVAL_COMMON_H
#define MCU_EVAL_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MCU_EVAL_CPU_HZ
#define MCU_EVAL_CPU_HZ 400000000u /* TC4D7 按 400 MHz 核估负载；500 MHz 改这个宏 */
#endif
#ifndef MCU_EVAL_DT_S
#define MCU_EVAL_DT_S 0.1f /* SoX 环 10 Hz */
#endif
#ifndef MCU_EVAL_N_CELL
#define MCU_EVAL_N_CELL 180 /* 800 V 包串数，和 Doc/A0-a 一致 */
#endif
#ifndef MCU_EVAL_LOG_BYTES
#define MCU_EVAL_LOG_BYTES (64u * 1024u) /* 串口要发的大 char 数组 */
#endif

#define MCU_EVAL_C1_STAR 2.8e4f
#define MCU_EVAL_R0_MIN 1.0e-5f
#define MCU_EVAL_R1_MIN 1.0e-5f
#define MCU_EVAL_DT 0.1f
#define MCU_EVAL_Q_COULOMB (100.0f * 3600.0f)

void McuEval_Init(void);
uint32_t McuEval_GetCycles(void);

void McuEval_LogClear(void);
void McuEval_LogPrintf(const char *fmt, ...);
char *McuEval_LogBuf(void);
unsigned McuEval_LogLen(void);

uint32_t McuEval_RngU32(void);
float McuEval_RngF(float scale);

float McuEval_Clampf(float x, float lo, float hi);
float McuEval_Softplus(float x);
float McuEval_Gelu(float x);
void McuEval_Ecm1(float i_a, float u_p, float r0, float r1, float c1, float u_ocv,
                  float *u_p_next, float *u_t, float *alpha);
float McuEval_Ocv(float s, float t_c);
float McuEval_DocvDs(float s, float t_c);

void McuEval_ReportHeader(const char *algo);
void McuEval_ReportCycles(const char *tag, unsigned n_iter, unsigned n_cell,
                          uint32_t cyc_min, unsigned long long cyc_sum);
void McuEval_ReportSize(const char *tag, unsigned bytes, const char *as_rom_or_ram);

#ifdef __cplusplus
}
#endif
#endif
