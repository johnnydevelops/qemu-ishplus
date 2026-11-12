/*
 * The per-CPU TranslationBlock jump cache.
 *
 *  Copyright (c) 2003 Fabrice Bellard
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef ACCEL_TCG_TB_JMP_CACHE_H
#define ACCEL_TCG_TB_JMP_CACHE_H

#include "qemu/rcu.h"
#include "exec/cpu-common.h"

/*
 * iSH+: 64K entries instead of 4K (1.5 MB per vCPU): the kernel's and the
 * programs' code missed in 4K, and every miss is a hash table lookup. A
 * whole flush (every guest context switch changes the ASID) only bumps
 * the generation, so a bigger cache costs nothing there.
 */
#define TB_JMP_CACHE_BITS 16
#define TB_JMP_CACHE_SIZE (1 << TB_JMP_CACHE_BITS)

/*
 * Invalidated in parallel; all accesses to 'tb' must be atomic.
 * A valid entry is read/written by a single CPU, therefore there is
 * no need for qatomic_rcu_read() and pc is always consistent with a
 * non-NULL value of 'tb'.  Strictly speaking pc is only needed for
 * CF_PCREL, but it's used always for simplicity.
 */
typedef struct CPUJumpCache {
    struct rcu_head rcu;
    /* iSH+: entries of other generations are empty (tcg_flush_jmp_cache) */
    uint64_t gen;
    /*
     * iSH+: what valid entries hold in .gen: the generation (high half),
     * and for the lower half of the address space (a process's code) the
     * address space's tag (low half, tb_jmp_cache_set_tag): a guest
     * process switch changes the tag, not the generation, and switching
     * back finds that process's entries again. [0] lower, [1] upper half.
     */
    uint64_t stamp[2];
    uint32_t tag;
    /* iSH+: and a generation of the lower half only (tcg_flush_jmp_cache_lower) */
    uint64_t lgen;
    struct {
        TranslationBlock *tb;
        vaddr pc;
        uint64_t gen;
    } array[TB_JMP_CACHE_SIZE];
} CPUJumpCache;

static inline uint64_t tb_jmp_cache_stamp(CPUJumpCache *jc, vaddr pc)
{
    return qatomic_read(&jc->stamp[(int64_t)pc < 0]);
}

/* After a change of gen or tag (on the vCPU's thread, or with all stopped) */
static inline void tb_jmp_cache_restamp(CPUJumpCache *jc)
{
    uint64_t g = qatomic_read(&jc->gen);

    qatomic_set(&jc->stamp[1], g << 32);
    /* gen + lgen grows with either: each of its values is used once */
    qatomic_set(&jc->stamp[0], (g + jc->lgen) << 32 | jc->tag);
}

#endif /* ACCEL_TCG_TB_JMP_CACHE_H */
