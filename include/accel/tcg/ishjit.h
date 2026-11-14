/*
 * iSH+ ishjit: AArch64-to-AArch64 translation of TBs (target/arm/tcg/ishjit.c)
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ACCEL_TCG_ISHJIT_H
#define ACCEL_TCG_ISHJIT_H

/* What the hook returns to leave the TB to TCG (-1: the code buffer is full) */
#define ISHJIT_DECLINE (-100)

/*
 * Set by a target that translates TBs itself: fills @buf (the TB's code, RW)
 * and tcg_ctx's per-insn data like setjmp_gen_code, returning the code size,
 * -1 or ISHJIT_DECLINE.
 */
extern int (*ishjit_gen_code_hook)(CPUState *cs, TranslationBlock *tb,
                                   vaddr pc, void *host_pc, int max_insns,
                                   void *buf);

/* The TCG epilogue's TB-exit entry (x0 = TB | exit index) */
const void *tcg_ishjit_tb_ret_addr(void);

/* debugging: the ISHJIT_PMAP file, or NULL */
FILE *ishjit_pmap_get(void);

void ishjit_enable(void *env);    /* (a CPUARMState) */
void ishjit_env_reset(void *env); /* (after a CPU reset cleared its fields) */

#endif
