#include "mcu_eval_common.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

char g_mcu_log[MCU_EVAL_LOG_BYTES];
static unsigned s_log_n;
static uint32_t s_rng;

void McuEval_Init(void)
{
    s_log_n = 0;
    if (MCU_EVAL_LOG_BYTES > 0u) {
        g_mcu_log[0] = '\0';
    }
    s_rng = 0xA5A5C3C3u;

#if defined(__TASKING__)
    /* CPU_CCTRL=0xFC00 打开时钟计数；CPU_CCNT=0xFC04 */
    __mtcr(0xFC00, 0x2);
#elif defined(__TRICORE__)
    /* AURIX GCC / HighTec：CSFR 必须用立即数约束 "i"，不能写 $FC04（会变成链接符号） */
    {
        unsigned v = 2u;
        __asm__ volatile("mtcr %0, %1" ::"i"(0xFC00), "d"(v) : "memory");
        __asm__ volatile("isync" ::: "memory");
    }
#endif
    McuEval_LogPrintf("mcu_eval init cpu_hz=%lu dt=%.3f n_cell=%u log_B=%u\n",
                      (unsigned long)MCU_EVAL_CPU_HZ, (double)MCU_EVAL_DT_S,
                      (unsigned)MCU_EVAL_N_CELL, (unsigned)MCU_EVAL_LOG_BYTES);
}

uint32_t McuEval_GetCycles(void)
{
#if defined(__TASKING__)
    return (uint32_t)__mfcr(0xFC04);
#elif defined(__TRICORE__)
    {
        unsigned v;
        __asm__ volatile("mfcr %0, %1" : "=d"(v) : "i"(0xFC04) : "memory");
        return (uint32_t)v;
    }
#elif defined(_MSC_VER)
    return (uint32_t)__rdtsc();
#elif defined(__GNUC__) && (defined(__i386__) || defined(__x86_64__))
    return (uint32_t)__builtin_ia32_rdtsc();
#else
    {
        static uint32_t fake;
        fake += 1u;
        return fake;
    }
#endif
}

void McuEval_LogClear(void)
{
    s_log_n = 0;
    if (MCU_EVAL_LOG_BYTES > 0u) {
        g_mcu_log[0] = '\0';
    }
}

void McuEval_LogPrintf(const char *fmt, ...)
{
    va_list ap;
    int n;
    unsigned room;

    if (s_log_n + 1u >= MCU_EVAL_LOG_BYTES) {
        return;
    }
    room = MCU_EVAL_LOG_BYTES - s_log_n;
    va_start(ap, fmt);
    n = vsnprintf(&g_mcu_log[s_log_n], room, fmt, ap);
    va_end(ap);
    if (n < 0) {
        return;
    }
    if ((unsigned)n >= room) {
        s_log_n = MCU_EVAL_LOG_BYTES - 1u;
    } else {
        s_log_n += (unsigned)n;
    }
    g_mcu_log[s_log_n] = '\0';
}

char *McuEval_LogBuf(void) { return g_mcu_log; }
unsigned McuEval_LogLen(void) { return s_log_n; }

uint32_t McuEval_RngU32(void)
{
    uint32_t x = s_rng;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_rng = x;
    return x;
}

float McuEval_RngF(float scale)
{
    /* [-scale, +scale] */
    float u = (float)(McuEval_RngU32() & 0xFFFFu) / 32767.5f - 1.0f;
    return u * scale;
}

float McuEval_Clampf(float x, float lo, float hi)
{
    if (!(x >= lo)) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

float McuEval_Softplus(float x)
{
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return logf(1.0f + expf(x));
}

float McuEval_Gelu(float x)
{
    /* 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) */
    float x3 = x * x * x;
    float u = 0.79788456f * (x + 0.044715f * x3);
    float t = tanhf(u);
    return 0.5f * x * (1.0f + t);
}

void McuEval_Ecm1(float i_a, float u_p, float r0, float r1, float c1, float u_ocv,
                  float *u_p_next, float *u_t, float *alpha)
{
    float tau = r1 * c1;
    float a;
    if (tau < 1.0e-6f) {
        tau = 1.0e-6f;
    }
    a = expf(-MCU_EVAL_DT / tau);
    *alpha = a;
    *u_p_next = a * u_p + r1 * (1.0f - a) * i_a;
    *u_t = u_ocv - i_a * r0 - *u_p_next;
}

/* 11 点 OCV，只为负荷真实一点（查表+插值），不是仓库 ocv.py 精度。 */
static const float k_ocv[11] = {
    3.20f, 3.42f, 3.55f, 3.62f, 3.67f, 3.70f, 3.78f, 3.90f, 4.02f, 4.12f, 4.20f};

float McuEval_Ocv(float s, float t_c)
{
    float x, f;
    int i;
    (void)t_c;
    s = McuEval_Clampf(s, 0.0f, 1.0f);
    x = s * 10.0f;
    i = (int)x;
    if (i < 0) {
        return k_ocv[0];
    }
    if (i >= 10) {
        return k_ocv[10];
    }
    f = x - (float)i;
    return k_ocv[i] * (1.0f - f) + k_ocv[i + 1] * f;
}

float McuEval_DocvDs(float s, float t_c)
{
    float lo = McuEval_Ocv(s - 0.01f, t_c);
    float hi = McuEval_Ocv(s + 0.01f, t_c);
    return (hi - lo) / 0.02f;
}

void McuEval_ReportHeader(const char *algo)
{
    McuEval_LogPrintf("---- algo=%s ----\n", algo);
}

void McuEval_ReportCycles(const char *tag, unsigned n_iter, unsigned n_cell,
                          uint32_t cyc_min, unsigned long long cyc_sum)
{
    double avg = (n_iter > 0u) ? (double)cyc_sum / (double)n_iter : 0.0;
    double budget = (double)MCU_EVAL_CPU_HZ * (double)MCU_EVAL_DT_S;
    double load = (budget > 0.0) ? (avg * (double)n_cell) / budget * 100.0 : 0.0;
    McuEval_LogPrintf(
        "%s n_iter=%u n_cell=%u cyc_min=%lu cyc_avg=%.1f "
        "load_pct_at_%uMHz_%.0fms=%.4f\n",
        tag, n_iter, n_cell, (unsigned long)cyc_min, avg, (unsigned)(MCU_EVAL_CPU_HZ / 1000000u),
        (double)MCU_EVAL_DT_S * 1000.0, load);
}

void McuEval_ReportSize(const char *tag, unsigned bytes, const char *as_rom_or_ram)
{
    McuEval_LogPrintf("%s_B=%u count_as=%s\n", tag, bytes, as_rom_or_ram);
}
