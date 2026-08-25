/* 主机或 Infineon 一次跑完全部 bench，日志进 g_mcu_log。
 * 工程里加入：mcu_eval_common.c 与除本文件 STANDALONE 以外的全部 eval_*.c。
 */
#include "mcu_eval_algos.h"
#include <stdio.h>

int main(void)
{
    McuEval_Init();
    McuEval_Ecm_Bench();
    McuEval_Ekf_Bench();
    McuEval_EkfDr0_Bench();
    McuEval_KGlobal_Bench();
    McuEval_KGrid_Bench();
    McuEval_PulseR0_Bench();
    McuEval_Head382_Bench();
    McuEval_Mlp16_Bench();
    McuEval_Lut2_Bench();
    McuEval_Lut3_Bench();
    McuEval_Mlp64_Bench();
    McuEval_IncrKGrid_Bench();
    McuEval_SoxCell_Bench();
    McuEval_SoxPack_Bench();
    McuEval_LogPrintf("---- done log_len=%u ----\n", McuEval_LogLen());

    fwrite(McuEval_LogBuf(), 1, McuEval_LogLen(), stdout);
    return 0;
}
