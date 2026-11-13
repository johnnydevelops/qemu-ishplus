/*
 * iSH+ ishjit: an AArch64-to-AArch64 translator for TCG translation blocks.
 *
 * Guest and host are both AArch64. Instead of decoding each guest instruction
 * into TCG ops, ishjit copies the instructions it knows to the host, with
 * their register numbers remapped (guest registers live in host registers,
 * guest NZCV in the host's NZCV). It produces a whole TB in QEMU's format
 * (code, per-insn unwind data, goto_tb patch points), so TB lookup, chaining,
 * interrupts, invalidation and cpu_restore_state all work as for TCG; any
 * block it can't translate is left to TCG (plan: docs/arm-translator-plan.md
 * in iSH+).
 *
 * Phase 1: integer data processing and branches. Phase 2: integer loads and
 * stores (inline softmmu TLB check, QEMU's helpers on a miss). FP/SIMD and
 * system instructions end the block (TCG translates from there).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/cacheflush.h"
#include "qemu/main-loop.h"
#include "accel/tcg/cpu-ops.h"
#include "cpu.h"
#include "hw/core/cpu.h"
#include "internals.h"
#include "exec/translation-block.h"
#include "tcg/tcg.h"
#include "tcg/insn-start-words.h"
#include "accel/tcg/ishjit.h"
#include "translate.h"
#include "cpregs.h"
#include "exec/address-spaces.h"

#include "ishjit-simd.h"
#include "accel/tcg/tb-jmp-cache.h"
#include "accel/tcg/tb-hash.h"
#include "tcg/tcg-ldst.h"
#include "exec/cpu_ldst.h"
#include "exec/memopidx.h"

/* From accel/tcg (tcg-runtime.h): the TB to continue with, or the epilogue */
const void *helper_lookup_tb_ptr(CPUArchState *env);
/* From target/arm (helper.h, helper-a64.h) */
void helper_exception_with_syndrome(CPUArchState *env, uint32_t excp, uint32_t syn);
void helper_exception_bkpt_insn(CPUArchState *env, uint32_t syn);
void helper_exception_return(CPUArchState *env, uint64_t new_pc);
void helper_msr_i_daifset(CPUArchState *env, uint32_t imm);
void helper_msr_i_daifclear(CPUArchState *env, uint32_t imm);

/*
 * Registers. Host x18 is Apple's (zeroed on context switches), host SP must
 * stay a host stack; x19 is env (TCG's TCG_AREG0, so TCG TBs and ishjit TBs
 * chain), x16, x17 and x27 are scratch (the TLB check needs three). The 26
 * others hold guest registers: x0-x12, x16, x17, x19-x26, x29, x30 and SP.
 * The six least used guest registers (measured over the Ubuntu workload:
 * x13, x14, x15, x18, x27, x28, together 1.5% of register accesses) stay in
 * env and are loaded into a scratch when used.
 */
#define H_ENV   19
#define H_TMP0  16
#define H_TMP1  17
#define H_TMP2  27
#define H_NONE  (-1)
/* guest registers kept in env */
#define ENV_HELD ((1u << 13) | (1u << 14) | (1u << 15) | (1u << 18) | \
                  (1u << 27) | (1u << 28))

/* guest register (0-30, 31 = SP) -> host register, or H_NONE (in env) */
static const int8_t g2h[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    H_NONE, H_NONE, H_NONE,             /* x13-x15 */
    13, 14,                             /* x16, x17 */
    H_NONE,                             /* x18 */
    15,                                 /* x19 */
    20, 21, 22, 23, 24, 25, 26,         /* x20-x26 */
    H_NONE,                             /* x27 */
    H_NONE,                             /* x28 */
    29, 30,                             /* x29, x30 */
    28,                                 /* SP */
};

/* How an instruction field holding 31 is read */
enum { F_NONE, F_ZR, F_SP };

typedef struct {
    uint8_t rd, rn, rm, ra;     /* F_* for bits [4:0], [9:5], [20:16], [14:10] */
    bool rd_written;
    bool reads_flags, writes_flags;
    uint32_t sysoff;            /* (mrs/msr of a plain field: its env offset) */
    bool spcheck;               /* (sp_el0: UNDEF while it is the stack pointer) */
    uint64_t sysconst;          /* (mrs of an ARM_CP_CONST register) */
} OperandInfo;

typedef struct {
    uint32_t *ptr;              /* where the next host insn goes (RW) */
    uint32_t *start;            /* the TB's code (RW) */
    uint8_t *highwater;
    TranslationBlock *tb;
    CPUARMState *env;
    vaddr pc_first;
    /* guest registers the block reads or writes (bit 31 = SP) */
    uint32_t used, written;
    bool flags_in, flags_out;   /* NZCV read before written; NZCV written */
    bool flags_written;         /* (so far, while scanning) */
    /* memory accesses: the TB's MMU index, TBI for data, alignment checks */
    int mmu_idx;
    int tbid;
    bool mem_ok;
    unsigned page_bits;
    int el;
    /* slow paths of the block's memory accesses */
    int nslow;
    /*
     * Guest V registers: in the host's V0-V31 within a TB (loaded after
     * the entry check, stored before every exit; not yet kept across TBs)
     */
    uint32_t vused, vwritten;
    bool fp_ok;                 /* FP/SIMD not trapped (FPEXC_EL 0) */
    bool fp_default;            /* FPCR all default: host FP is the guest's */
    bool fgt_active, fgt_svc, trap_eret;
    bool unpriv;                /* TB flag UNPRIV: ldtr/sttr use the user index */
    /* shared TLB checks (emit_mem): this access's role, its group's slot and span */
    int g_role;                 /* 0 none, 1 the group's first, 2 a later one */
    int g_slot, g_min, g_span;
    bool g_store;
    bool el2_el3;               /* (the CPU has EL2 or EL3: more access checks) */
    uint64_t sysconst;          /* (classify_sys: an ARM_CP_CONST's value) */
    uint32_t cur_insn;          /* the guest insn being emitted */
    /* (branches to the TB's jumps to R_PCSYNC, R_EXITTB, emitted at its end) */
    uint32_t *pend_pcsync[8], *pend_exit[8];
    int npcsync, nexit;
    /*
     * Cold code (slow paths, exit paths, the full entry) goes to a separate
     * buffer while translating, then below the TB's struct at the region's
     * top, away from the hot code: branches between the two are fixed up
     */
    bool in_cold;
    uint32_t *hot_save, *cptr;
    uint32_t *jmp2_c[2], *jmp2r_c[2];
} IJ;

#define IJ_COLD_MAX (64 * 1024 / 4)
#define IJ_HIGHWATER 1024   /* (TCG's TCG_HIGHWATER) */
static __thread uint32_t ij_cold_buf[IJ_COLD_MAX + 1024];
enum { FX_H2C, FX_C2H, FX_C2H_REL };
typedef struct { int kind; uint32_t *at, *to, *base; } IJFix;
#define IJ_NFIX 4096
static __thread IJFix ij_fix[IJ_NFIX];
static __thread int ij_nfix;

static void ij_to_cold(IJ *j)
{
    g_assert(!j->in_cold);
    j->hot_save = j->ptr;
    j->ptr = j->cptr;
    j->in_cold = true;
}

static void ij_to_hot(IJ *j)
{
    g_assert(j->in_cold);
    j->cptr = j->ptr;
    j->ptr = j->hot_save;
    j->in_cold = false;
}

static void ij_fixup(int kind, uint32_t *at, uint32_t *to, uint32_t *base)
{
    g_assert(ij_nfix < IJ_NFIX);
    ij_fix[ij_nfix++] = (IJFix){ kind, at, to, base };
}


static inline void emit(IJ *j, uint32_t insn)
{
    *j->ptr++ = insn;
}

/* (hot code) b @cold */
static void emit_b_cold(IJ *j, uint32_t *cold)
{
    g_assert(!j->in_cold);
    ij_fixup(FX_H2C, j->ptr, cold, NULL);
    emit(j, 0);
}

static inline intptr_t ij_offset(IJ *j)
{
    return (uint8_t *)j->ptr - (uint8_t *)j->start;
}

/* debugging (ISHJIT_PMAP): kinds of code regions in a TB, for profiles */
static __thread struct { uint32_t o0, o1; char k; } ij_reg[4096];
static __thread int ij_nreg;
static inline void ij_region(IJ *j, intptr_t o0, char k)
{
    if (ij_nreg < 4096 && !j->in_cold) {
        ij_reg[ij_nreg].o0 = o0;
        ij_reg[ij_nreg].o1 = ij_offset(j);
        ij_reg[ij_nreg].k = k;
        ij_nreg++;
    }
}

/* Host instruction encodings (64-bit forms unless noted) */
static uint32_t i_ldr_x(int rt, int rn, unsigned off)
{
    return 0xF9400000 | (off / 8) << 10 | rn << 5 | rt;
}
static uint32_t i_str_x(int rt, int rn, unsigned off)
{
    return 0xF9000000 | (off / 8) << 10 | rn << 5 | rt;
}
static uint32_t i_ldr_w(int rt, int rn, unsigned off)
{
    return 0xB9400000 | (off / 4) << 10 | rn << 5 | rt;
}
static uint32_t i_str_w(int rt, int rn, unsigned off)
{
    return 0xB9000000 | (off / 4) << 10 | rn << 5 | rt;
}
static uint32_t i_ldur_w(int rt, int rn, int off)
{
    return 0xB8400000 | (off & 0x1ff) << 12 | rn << 5 | rt;
}
static uint32_t i_mov_x(int rd, int rm)     /* orr rd, xzr, rm */
{
    return 0xAA0003E0 | rm << 16 | rd;
}
static uint32_t i_br(int rn)
{
    return 0xD61F0000 | rn << 5;
}
static uint32_t i_blr(int rn)
{
    return 0xD63F0000 | rn << 5;
}
static uint32_t i_b(intptr_t insns)
{
    return 0x14000000 | (insns & 0x3ffffff);
}
static uint32_t i_adr(int rd, intptr_t bytes)
{
    return 0x10000000 | (bytes & 3) << 29 | ((bytes >> 2) & 0x7ffff) << 5 | rd;
}

static void emit_movi(IJ *j, int rd, uint64_t v)
{
    bool first = true;

    for (int hw = 0; hw < 4; hw++) {
        uint32_t part = (v >> (hw * 16)) & 0xffff;

        if (part || (first && hw == 3)) {
            emit(j, (first ? 0xD2800000 : 0xF2800000) | hw << 21 | part << 5 | rd);
            first = false;
        }
    }
    if (first) {
        emit(j, 0xD2800000 | rd);   /* movz rd, #0 */
    }
}

/* rd = rn + v (any 64-bit v; uses H_TMP1 when v does not fit 12 bits) */
static void emit_addi(IJ *j, int rd, int rn, int64_t v)
{
    uint64_t a = v < 0 ? -(uint64_t)v : v;
    uint32_t op = v < 0 ? 0xD1000000 : 0x91000000;     /* sub / add imm */

    if (a < 4096) {
        emit(j, op | a << 10 | rn << 5 | rd);
    } else if ((a & 0xfff) == 0 && a < (1 << 24)) {
        emit(j, op | 1 << 22 | (a >> 12) << 10 | rn << 5 | rd);
    } else if (a < (1 << 24)) {
        emit(j, op | 1 << 22 | (a >> 12) << 10 | rn << 5 | rd);
        emit(j, op | (a & 0xfff) << 10 | rd << 5 | rd);
    } else {
        emit_movi(j, H_TMP1, v);
        emit(j, 0x8B000000 | H_TMP1 << 16 | rn << 5 | rd);  /* add rd, rn, x17 */
    }
}

static unsigned xreg_off(int g)
{
    return offsetof(CPUARMState, xregs[g]);
}

static unsigned vreg_off(int v)
{
    return offsetof(CPUARMState, vfp.zregs[v]);
}

static int tlb_fast_off(int mmu_idx)
{
    return (int)offsetof(ArchCPU, parent_obj.neg.tlb.f[mmu_idx]) -
           (int)offsetof(ArchCPU, env);
}


/*
 * A64 logical immediates: N:immr:imms is reserved when no element size
 * results (N = 0, imms = 11111x) or the element's bits are all ones
 * (DecodeBitMasks); such encodings are UNDEFINED, left to TCG.
 */
static bool logic_imm_ok(uint32_t insn, bool sf)
{
    unsigned n = (insn >> 22) & 1, imms = (insn >> 10) & 0x3f;
    unsigned v = n << 6 | (~imms & 0x3f);
    int len = v ? 31 - clz32(v) : -1;

    if (!sf && n) {
        return false;
    }
    if (len < 1) {
        return false;
    }
    /* S (imms within the element) must not be all ones */
    return (imms & ((1u << len) - 1)) != (1u << len) - 1;
}

/*
 * Operand roles of the instructions ishjit copies (A64 "Data Processing --
 * Immediate" and "-- Register" classes, the A13's subset), or false.
 */
static bool decode_dp(uint32_t insn, OperandInfo *o)
{
    memset(o, 0, sizeof(*o));
    o->rd_written = true;

    if ((insn & 0x1C000000) == 0x10000000) {        /* DP immediate */
        switch ((insn >> 23) & 7) {
        case 2:                                     /* add/sub immediate */
            o->rn = F_SP;
            o->rd = (insn & (1u << 29)) ? F_ZR : F_SP;
            o->writes_flags = insn & (1u << 29);
            return true;
        case 4:                                     /* logical immediate */
            if (!logic_imm_ok(insn, insn & (1u << 31))) {
                return false;
            }
            o->rn = F_ZR;
            o->rd = ((insn >> 29) & 3) == 3 ? F_ZR : F_SP;
            o->writes_flags = ((insn >> 29) & 3) == 3;
            return true;
        case 5:                                     /* move wide */
            if (((insn >> 29) & 3) == 1) {
                return false;
            }
            if (!(insn & (1u << 31)) && (insn & (1u << 22))) {
                return false;                       /* hw >= 2 with 32 bits */
            }
            o->rd = F_ZR;
            if (((insn >> 29) & 3) == 3) {
                o->rn = F_NONE;                     /* movk: reads rd */
            }
            return true;
        case 6:                                     /* bitfield */
            if (((insn >> 29) & 3) == 3 ||
                ((insn >> 31) & 1) != ((insn >> 22) & 1) ||
                (!(insn & (1u << 31)) &&
                 ((insn & (1u << 21)) || (insn & (1u << 15))))) {
                return false;                       /* (immr, imms < 32) */
            }
            o->rd = F_ZR;
            o->rn = F_ZR;
            return true;
        case 7:                                     /* extract */
            if ((insn & 0x60200000) ||
                ((insn >> 31) & 1) != ((insn >> 22) & 1) ||
                (!(insn & (1u << 31)) && (insn & (1u << 15)))) {
                return false;
            }
            o->rd = o->rn = o->rm = F_ZR;
            return true;
        default:                                    /* adr: handled apart */
            return false;
        }
    }

    if ((insn & 0x0E000000) != 0x0A000000) {
        return false;
    }
    /* DP register */
    if ((insn & 0x1F000000) == 0x0A000000) {        /* logical shifted reg */
        if (!(insn & (1u << 31)) && (insn & (1u << 15))) {
            return false;
        }
        o->rd = o->rn = o->rm = F_ZR;
        o->writes_flags = ((insn >> 29) & 3) == 3;
        return true;
    }
    if ((insn & 0x1F200000) == 0x0B000000) {        /* add/sub shifted reg */
        if (((insn >> 22) & 3) == 3 ||
            (!(insn & (1u << 31)) && (insn & (1u << 15)))) {
            return false;
        }
        o->rd = o->rn = o->rm = F_ZR;
        o->writes_flags = insn & (1u << 29);
        return true;
    }
    if ((insn & 0x1F200000) == 0x0B200000) {        /* add/sub extended reg */
        if ((insn & 0x00C00000) || ((insn >> 10) & 7) > 4) {
            return false;
        }
        o->rn = F_SP;
        o->rm = F_ZR;
        o->rd = (insn & (1u << 29)) ? F_ZR : F_SP;
        o->writes_flags = insn & (1u << 29);
        return true;
    }
    if ((insn & 0x1FE00000) == 0x1A000000) {        /* adc/sbc(s) */
        if (insn & 0x0000FC00) {
            return false;                           /* (rmif, setf) */
        }
        o->rd = o->rn = o->rm = F_ZR;
        o->reads_flags = true;
        o->writes_flags = insn & (1u << 29);
        return true;
    }
    if ((insn & 0x1FE00000) == 0x1A400000) {        /* ccmn/ccmp */
        if (!(insn & (1u << 29)) || (insn & 0x00000410)) {
            return false;
        }
        o->rd_written = false;
        o->rd = F_NONE;
        o->rn = F_ZR;
        o->rm = (insn & (1u << 11)) ? F_NONE : F_ZR;  /* immediate form */
        o->reads_flags = o->writes_flags = true;
        return true;
    }
    if ((insn & 0x1FE00000) == 0x1A800000) {        /* csel family */
        if ((insn & (1u << 29)) || (insn & 0x00000800)) {
            return false;
        }
        o->rd = o->rn = o->rm = F_ZR;
        o->reads_flags = true;
        return true;
    }
    if ((insn & 0x1F000000) == 0x1B000000) {        /* 3-source */
        uint32_t op = ((insn >> 21) & 7) << 1 | ((insn >> 15) & 1);
        bool sf = insn & (1u << 31);

        if ((insn >> 29) & 3) {
            return false;
        }
        switch (op) {
        case 0: case 1:                             /* madd, msub */
            break;
        case 2: case 3: case 10: case 11:           /* s/umaddl, s/umsubl */
        case 4: case 12:                            /* smulh, umulh */
            if (!sf) {
                return false;
            }
            break;
        default:
            return false;
        }
        o->rd = o->rn = o->rm = F_ZR;
        o->ra = (op == 4 || op == 12) ? F_NONE : F_ZR;
        return true;
    }
    if ((insn & 0x5FE00000) == 0x1AC00000) {        /* 2-source */
        uint32_t op = (insn >> 10) & 0x3f;

        if (insn & (1u << 29)) {
            return false;
        }
        switch (op) {
        case 2: case 3:                             /* udiv, sdiv */
        case 8: case 9: case 10: case 11:           /* lslv... rorv */
            break;
        case 16: case 17: case 18: case 20: case 21: case 22: /* crc32(c) b/h/w */
            if (insn & (1u << 31)) {
                return false;
            }
            break;
        case 19: case 23:                           /* crc32(c)x */
            if (!(insn & (1u << 31))) {
                return false;
            }
            break;
        default:
            return false;
        }
        o->rd = o->rn = o->rm = F_ZR;
        return true;
    }
    if ((insn & 0x5FE00000) == 0x5AC00000) {        /* 1-source */
        uint32_t op = (insn >> 10) & 0x3f;

        if ((insn & (1u << 29)) || ((insn >> 16) & 0x1f)) {
            return false;
        }
        switch (op) {
        case 0: case 1: case 2: case 4: case 5:     /* rbit rev16 rev(32) clz cls */
            break;
        case 3:                                     /* rev (64) */
            if (!(insn & (1u << 31))) {
                return false;
            }
            break;
        default:
            return false;
        }
        o->rd = o->rn = F_ZR;
        return true;
    }
    return false;
}

/*
 * Rewrites a copied instruction's registers. Fields reading SP take the
 * host register holding the guest SP; those reading ZR keep 31; guest
 * registers kept in env use the scratch registers given in @tmp.
 */
static int host_reg_for(int g, int role, const int *tmp)
{
    if (g == 31) {
        return role == F_SP ? g2h[31] : 31;
    }
    if (g2h[g] == H_NONE) {
        return tmp[g];
    }
    return g2h[g];
}

/* guest register numbers an instruction reads / writes (31 = SP only) */
static void operand_regs(uint32_t insn, const OperandInfo *o,
                         uint32_t *reads, uint32_t *writes)
{
    int rd = insn & 31, rn = (insn >> 5) & 31;
    int rm = (insn >> 16) & 31, ra = (insn >> 10) & 31;

    *reads = *writes = 0;
#define RD_FIELD(f, r) \
    if ((f) != F_NONE && ((r) != 31 || (f) == F_SP)) { *reads |= 1u << (r); }
    RD_FIELD(o->rn, rn);
    RD_FIELD(o->rm, rm);
    RD_FIELD(o->ra, ra);
#undef RD_FIELD
    if (o->rd != F_NONE && (rd != 31 || o->rd == F_SP)) {
        if (o->rd_written) {
            *writes |= 1u << rd;
        }
        /* movk and bfm keep the other bits of rd */
        if ((insn & 0x7F800000) == 0x72800000 ||
            (insn & 0x7F800000) == 0x33000000) {
            *reads |= 1u << rd;
        }
    }
}

static void emit_flags_from_env(IJ *j);
static void emit_flags_to_env(IJ *j);


/*
 * Shared routines (generated once, after TCG's prologue, so tb_flush keeps
 * them), instead of copies in every TB: called with the return address in
 * x27, through their address in env->ishjit_routine[] (any distance).
 *   R_LOAD: NZCV and every guest register from env (a TB's full entry)
 *   R_SYNC: every guest register and NZCV to env (leaving ishjit code)
 *   R_SAVE: as R_SYNC, and x1 = x16 (a slow path's address)
 *   [3]: a per-vCPU scratch slot
 */
enum { ISHJIT_R_LOAD, ISHJIT_R_SYNC, ISHJIT_R_SLOW, ISHJIT_R_SCRATCH,
       ISHJIT_R_LOOKUP, ISHJIT_R_PCSYNC, ISHJIT_R_EXITTB, ISHJIT_NROUTINES = 8 };
static unsigned shared_page_bits;
static void emit_lookup_body(IJ *j, bool shared);
static uint64_t ishjit_routine_rx[ISHJIT_NROUTINES];
typedef struct { uint64_t cont, vthunk; } IJSlowRet;
IJSlowRet ishjit_slow_c(CPUARMState *env, uint64_t addr, const uint32_t *desc);

/*
 * debugging: ISHJIT_SITES=1 records where full stores (R_SYNC) are called
 * from, and prints the top call sites at exit with the TB's guest end
 */
#define ISHJIT_SITES_N (1 << 20)
static uint64_t ishjit_site_idx;
static uint64_t *ishjit_sites;
static int site_cmp(const void *a, const void *b)
{
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}
static void ishjit_sites_dump(void)
{
    uint64_t n = MIN(ishjit_site_idx, ISHJIT_SITES_N);
    uint64_t top_a[20] = {0}, top_c[20] = {0};

    qsort(ishjit_sites, n, 8, site_cmp);
    for (uint64_t i = 0; i < n;) {
        uint64_t k = i;
        while (k < n && ishjit_sites[k] == ishjit_sites[i]) {
            k++;
        }
        for (int t = 0; t < 20; t++) {
            if (k - i > top_c[t]) {
                memmove(&top_c[t + 1], &top_c[t], (19 - t) * 8);
                memmove(&top_a[t + 1], &top_a[t], (19 - t) * 8);
                top_c[t] = k - i;
                top_a[t] = ishjit_sites[i];
                break;
            }
        }
        i = k;
    }
    for (int t = 0; t < 20 && top_c[t]; t++) {
        TranslationBlock *tb = tcg_tb_lookup(top_a[t]);
        uint64_t paddr = tb ? tb_page_addr0(tb) : 0;
        uint32_t next = 0;

        if (tb && tb_page_addr0(tb) != -1) {
            next = ldl_le_phys(&address_space_memory, 0x40000000 + paddr + tb->size);  /* (virt: RAM base) */
        }
        uint32_t lastw = tb && tb->size ?
            ldl_le_phys(&address_space_memory, 0x40000000 + paddr + tb->size - 4) : 0;
        fprintf(stderr, "ISHJIT site %3d%% n=%llu tb pa=%llx size=%u last %08x next %08x\n",
                (int)(100 * top_c[t] / n), (unsigned long long)top_c[t],
                (unsigned long long)paddr, tb ? tb->size : 0, lastw, next);
    }
}

/* debugging: ISHJIT_COUNT=1 counts full loads and stores (printed at exit) */
static uint64_t ishjit_count[2];
static void ishjit_count_dump(void)
{
    fprintf(stderr, "ISHJIT full loads %llu, full stores %llu\n",
            (unsigned long long)ishjit_count[0], (unsigned long long)ishjit_count[1]);
}
static void emit_count(IJ *j, int k)
{
    if (getenv("ISHJIT_COUNT")) {
        emit_movi(j, H_TMP0, (uintptr_t)&ishjit_count[k]);
        emit(j, i_ldr_x(H_TMP1, H_TMP0, 0));
        emit(j, 0x91000400 | H_TMP1 << 5 | H_TMP1);             /* add x17, x17, #1 */
        emit(j, i_str_x(H_TMP1, H_TMP0, 0));
    }
}

static unsigned routine_off(int r);
static void emit_env_far(IJ *j, bool load, int rt, int tmp, unsigned off);

static void *ishjit_emit_shared(void *code_ptr)
{
    IJ jj = { .ptr = code_ptr, .start = code_ptr };
    IJ *j = &jj;

    if (getenv("ISHJIT_COUNT")) {
        atexit(ishjit_count_dump);
    }
    ishjit_routine_rx[ISHJIT_R_LOAD] = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    emit_count(j, 0);
    emit_flags_from_env(j);
    for (int g = 0; g < 32; g++) {
        if (g2h[g] != H_NONE) {
            emit(j, i_ldr_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
    emit(j, i_br(H_TMP2));

    for (int r = ISHJIT_R_SYNC; r <= ISHJIT_R_SLOW; r++) {
        ishjit_routine_rx[r] = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
        if (r == ISHJIT_R_SYNC) {
            emit_count(j, 1);
            if (getenv("ISHJIT_SITES")) {
                if (!ishjit_sites) {
                    ishjit_sites = g_new0(uint64_t, ISHJIT_SITES_N);
                    atexit(ishjit_sites_dump);
                }
                emit_movi(j, H_TMP0, (uintptr_t)&ishjit_site_idx);
                emit(j, i_ldr_x(H_TMP1, H_TMP0, 0));
                emit(j, 0x91000400 | H_TMP1 << 5 | H_TMP1);         /* add x17, x17, #1 */
                emit(j, i_str_x(H_TMP1, H_TMP0, 0));
                emit(j, 0x92404E31);                                /* and x17, x17, #0xfffff */
                emit_movi(j, H_TMP0, (uintptr_t)ishjit_sites);
                emit(j, 0xF8317A1B);                                /* str x27, [x16, x17, lsl #3] */
            }
        }
        for (int g = 0; g < 32; g++) {
            if (g2h[g] != H_NONE) {
                emit(j, i_str_x(g2h[g], H_ENV, xreg_off(g)));
            }
        }
        if (r == ISHJIT_R_SLOW) {
            /*
             * A memory access's slow path: x16 its address, x27 its
             * descriptor (emit_slow). Everything stored, ishjit_slow_c
             * does the access through QEMU's helpers and returns where to
             * go on (x0) and the TB's V register reload or 0 (x1); then
             * everything loaded again.
             */
            uint32_t *z;

            emit(j, i_mov_x(1, H_TMP0));
            emit(j, i_mov_x(2, H_TMP2));
            emit_flags_to_env(j);
            emit(j, i_mov_x(0, H_ENV));
            emit_movi(j, H_TMP0, (uintptr_t)ishjit_slow_c);
            emit(j, i_blr(H_TMP0));
            emit(j, i_mov_x(H_TMP2, 0));
            emit_env_far(j, false, 1, H_TMP0, routine_off(ISHJIT_R_SCRATCH));
            emit_flags_from_env(j);
            for (int g = 0; g < 32; g++) {
                if (g2h[g] != H_NONE) {
                    emit(j, i_ldr_x(g2h[g], H_ENV, xreg_off(g)));
                }
            }
            emit_env_far(j, true, H_TMP1, H_TMP1, routine_off(ISHJIT_R_SCRATCH));
            z = j->ptr;
            emit(j, 0);                                 /* cbz x17, (br x27) */
            emit(j, i_br(H_TMP1));
            *z = 0xB4000000 | ((j->ptr - z) & 0x7ffff) << 5 | H_TMP1;
            emit(j, i_br(H_TMP2));
            break;
        }
        emit_flags_to_env(j);
        emit(j, i_br(H_TMP2));
    }
    /*
     * R_LOOKUP: to the guest pc in x16 from the TB x27 (V registers
     * stored); R_PCSYNC: env->pc's page offset := x17, then as R_SYNC
     */
    shared_page_bits = j->page_bits = TARGET_PAGE_BITS;
    ishjit_routine_rx[ISHJIT_R_LOOKUP] = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    emit_lookup_body(j, true);
    ishjit_routine_rx[ISHJIT_R_PCSYNC] = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
    emit(j, 0x92400000 | (64 - j->page_bits) << 16 | (63 - j->page_bits) << 10 |
            H_TMP0 << 5 | H_TMP0);                  /* and x16, x16, #page mask */
    emit(j, 0xAA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);  /* orr x16, x16, x17 */
    emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
    emit(j, i_b(((uint32_t *)code_ptr + (ishjit_routine_rx[ISHJIT_R_SYNC] -
                  ishjit_routine_rx[ISHJIT_R_LOAD]) / 4) - j->ptr));
    return j->ptr;
}

static void __attribute__((constructor)) ishjit_register(void)
{
    tcg_extra_prologue_hook = ishjit_emit_shared;
}

static unsigned routine_off(int r)
{
    return offsetof(CPUARMState, ishjit_routine[r]);
}

/*
 * rt = / := *(u64 *)(env + off) for any offset (the scaled 12-bit immediate
 * reaches 32KB; the routines' slots are further, after the reset fields).
 * @tmp holds the address when needed (rt itself for a load).
 */
static void emit_env_far(IJ *j, bool load, int rt, int tmp, unsigned off)
{
    g_assert(off % 8 == 0 && off < (1u << 24));
    if (off / 8 <= 4095) {
        emit(j, load ? i_ldr_x(rt, H_ENV, off) : i_str_x(rt, H_ENV, off));
        return;
    }
    emit(j, 0x91400000 | (off >> 12) << 10 | H_ENV << 5 | tmp);   /* add tmp, env, #hi, lsl #12 */
    emit(j, load ? i_ldr_x(rt, tmp, off & 0xfff) : i_str_x(rt, tmp, off & 0xfff));
}

/* @via = R(r)'s address, then br @via (any distance) */
static void emit_jump_routine(IJ *j, int r, int via)
{
    emit_env_far(j, true, via, via, routine_off(r));
    emit(j, i_br(via));
}

/* call a shared routine (through @via: x16, or x17 when x16 is in use) */
static void emit_call_routine(IJ *j, int r, int via)
{
    uint32_t *adr = j->ptr;

    emit(j, 0);                                     /* adr x27, (after the br) */
    emit_env_far(j, true, via, via, routine_off(r));
    emit(j, i_br(via));
    *adr = i_adr(H_TMP2, (j->ptr - adr) * 4);
}

/* Loads of the guest registers the block uses; NZCV if read first */
static void emit_block_entry(IJ *j)
{
    /*
     * Phase 4: between ishjit TBs chained to each other, every guest
     * register keeps its host register and NZCV the host's NZCV; a TB
     * entered otherwise (from cpu_exec or a TCG TB) loads them all.
     */
    emit_call_routine(j, ISHJIT_R_LOAD, H_TMP0);
}

/* QEMU's NF/ZF/CF/VF to the host NZCV (uses x16, x17 and the flags) */
static void emit_flags_from_env(IJ *j)
{
    {
        emit(j, i_ldr_w(H_TMP0, H_ENV, offsetof(CPUARMState, NF)));
        emit(j, 0x12010000 | H_TMP0 << 5 | H_TMP0);         /* and w16, w16, #0x80000000 */
        emit(j, i_ldr_w(H_TMP1, H_ENV, offsetof(CPUARMState, ZF)));
        emit(j, 0x7100001F | H_TMP1 << 5);                  /* cmp w17, #0 */
        emit(j, 0x1A9F17E0 | H_TMP1);                       /* cset w17, eq */
        emit(j, 0x2A000000 | H_TMP1 << 16 | 30 << 10 | H_TMP0 << 5 | H_TMP0);
        emit(j, i_ldr_w(H_TMP1, H_ENV, offsetof(CPUARMState, CF)));
        emit(j, 0x2A000000 | H_TMP1 << 16 | 29 << 10 | H_TMP0 << 5 | H_TMP0);
        emit(j, i_ldr_w(H_TMP1, H_ENV, offsetof(CPUARMState, VF)));
        emit(j, 0x53000000 | 31 << 16 | 31 << 10 | H_TMP1 << 5 | H_TMP1); /* lsr w17, w17, #31 */
        emit(j, 0x2A000000 | H_TMP1 << 16 | 28 << 10 | H_TMP0 << 5 | H_TMP0);
        emit(j, 0xD51B4200 | H_TMP0);                       /* msr nzcv, x16 */
    }
}

/*
 * Stores of what the block wrote, without touching the host flags (a
 * conditional branch may follow).
 */
static void emit_sync(IJ *j)
{
    emit_call_routine(j, ISHJIT_R_SYNC, H_TMP0);
}

/* The host NZCV to QEMU's NF/ZF/CF/VF, flags untouched (uses x16, x17) */
static void emit_flags_to_env(IJ *j)
{
    {
        emit(j, 0xD53B4200 | H_TMP0);                       /* mrs x16, nzcv */
        emit(j, 0x12010000 | H_TMP0 << 5 | H_TMP1);         /* and w17, w16, #0x80000000 */
        emit(j, i_str_w(H_TMP1, H_ENV, offsetof(CPUARMState, NF)));
        emit(j, 0x53000000 | 30 << 16 | 30 << 10 | H_TMP0 << 5 | H_TMP1); /* ubfx w17, w16, #30, #1 */
        emit(j, 0x52000000 | H_TMP1 << 5 | H_TMP1);         /* eor w17, w17, #1 */
        emit(j, i_str_w(H_TMP1, H_ENV, offsetof(CPUARMState, ZF)));
        emit(j, 0x53000000 | 29 << 16 | 29 << 10 | H_TMP0 << 5 | H_TMP1); /* ubfx w17, w16, #29, #1 */
        emit(j, i_str_w(H_TMP1, H_ENV, offsetof(CPUARMState, CF)));
        emit(j, 0x53000000 | 29 << 16 | 28 << 10 | H_TMP0 << 5 | H_TMP1); /* lsl w17, w16, #3 */
        emit(j, i_str_w(H_TMP1, H_ENV, offsetof(CPUARMState, VF)));
    }
}


/* debugging: ISHJIT_CHECKENV=1 traps (brk #tag) where x19 is not env */
static int checkenv = -1;
static void emit_checkenv(IJ *j, int tag)
{
    if (checkenv < 0) {
        checkenv = getenv("ISHJIT_CHECKENV") != NULL;
    }
    if (checkenv) {
        /* flag-free: x17 = env ^ x19; cbz ok; ... for each vCPU; brk */
        CPUState *c;
        uint32_t *ok[16];
        int n = 0;

        CPU_FOREACH(c) {
            if (n < 16) {
                emit_movi(j, H_TMP1, (uintptr_t)cpu_env(c));
                emit(j, 0xCA000000 | H_ENV << 16 | H_TMP1 << 5 | H_TMP1);
                ok[n++] = j->ptr;
                emit(j, 0xB4000000 | H_TMP1);               /* cbz x17, ok */
            }
        }
        emit(j, 0xD4200000 | (tag & 0xffff) << 5);          /* brk #tag */
        for (int k = 0; k < n; k++) {
            *ok[k] |= ((j->ptr - ok[k]) & 0x7ffff) << 5;
        }
    }
}

/*
 * The guest pc. TBs are CF_PCREL, and within ishjit code only the page of
 * env->pc is kept up to date: jumps between TBs of one page (chained) leave
 * it alone, so its offset bits can be stale. restore_state_to_opc only
 * uses the page; whatever leaves ishjit code for C or TCG code sets the
 * exact value first (emit_pc_set).
 */
/* rd = the guest address @target (uses x17 for far ones, unless rd is x17) */
static void emit_pc_rel(IJ *j, int rd, vaddr target)
{
    unsigned pb = j->page_bits;

    emit(j, i_ldr_x(rd, H_ENV, offsetof(CPUARMState, pc)));
    /* and rd, rd, #~(page size - 1) */
    emit(j, 0x92400000 | (64 - pb) << 16 | (63 - pb) << 10 | rd << 5 | rd);
    emit_addi(j, rd, rd, (int64_t)(target - (j->pc_first & TARGET_PAGE_MASK)));
}

/* env->pc = @target (uses x16, x17) */
static void emit_pc_set(IJ *j, vaddr target)
{
    emit_pc_rel(j, H_TMP0, target);
    emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
}

/* rd = this TB's struct (RX address) + @add: adr when in reach, else movz/movk */
static void emit_tb_addr(IJ *j, int rd, unsigned add)
{
    intptr_t d = (uint8_t *)j->tb + add - (uint8_t *)j->ptr;

    if (!j->in_cold && d == sextract64(d, 0, 21)) {
        emit(j, i_adr(rd, d));
    } else {
        emit_movi(j, rd, (uintptr_t)tcg_splitwx_to_rx(j->tb) + add);
    }
}

static void emit_exit_tb(IJ *j, unsigned idx)
{
    const void *ret = tcg_ishjit_tb_ret_addr();
    uintptr_t here;
    ptrdiff_t off;

    /* x0 = the TB | idx */
    emit_tb_addr(j, 0, idx);
    here = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    off = ((uintptr_t)ret - here) >> 2;
    if (!j->in_cold && off == sextract64(off, 0, 26)) {
        emit(j, 0x14000000 | (off & 0x3ffffff));
    } else {
        g_assert(j->nexit < 8);
        j->pend_exit[j->nexit++] = j->ptr;
        emit(j, 0);                             /* b (the TB's exit stub) */
    }
}

/* env->pc := @target (in this page), registers stored, then on after this */
static void emit_pcsync(IJ *j, vaddr target)
{
    emit(j, 0xD2800000 | (uint32_t)(target & ~TARGET_PAGE_MASK) << 5 | H_TMP1);  /* movz x17, #off */
    emit(j, i_adr(H_TMP2, 8));
    g_assert(j->npcsync < 8);
    j->pend_pcsync[j->npcsync++] = j->ptr;
    emit(j, 0);                                 /* b (the TB's R_PCSYNC jump) */
}

/*
 * A chainable jump to @dest (in this TB's page). The first patch point goes
 * to the next TB's fast entry when it is an ishjit TB (registers in place,
 * env->pc's page right); unchained, or to a TCG TB, the pc is set and the
 * registers stored first, then the second patch point goes to that TB, or
 * exits to cpu_exec.
 */
static void emit_goto_tb(IJ *j, int which, vaddr dest)
{
    emit_checkenv(j, 2);
    j->tb->jmp_insn_offset[which] = ij_offset(j);
    emit(j, 0x14000000);                    /* b . (patched) */
    emit(j, i_br(H_TMP0));                  /* (when patched to ldr x16) */
    j->tb->jmp_reset_offset[which] = ij_offset(j);
    {
        /* (the unchained way: cold) */
        uint32_t *c;

        ij_to_cold(j);
        c = j->ptr;
        emit_pcsync(j, dest);
        j->jmp2_c[which] = j->ptr;
        emit(j, i_b(1));                    /* b (patched: a TCG TB) */
        j->jmp2r_c[which] = j->ptr;
        emit_exit_tb(j, which);
        ij_to_hot(j);
        emit_b_cold(j, c);
    }
}



static void emit_vload(IJ *j, uint32_t set)
{
    for (int v = 0; v < 32; v++) {
        if ((set >> v) & 1) {
            emit(j, 0x3DC00000 | (vreg_off(v) / 16) << 10 | H_ENV << 5 | v);   /* ldr qv */
        }
    }
}

static void emit_vstore(IJ *j, uint32_t set)
{
    for (int v = 0; v < 32; v++) {
        if ((set >> v) & 1) {
            emit(j, 0x3D800000 | (vreg_off(v) / 16) << 10 | H_ENV << 5 | v);   /* str qv */
        }
    }
}

/* rt = *(env + off), off negative (CPUState fields before env) */
static void emit_ld_env_neg(IJ *j, int rt, int off)
{
    if (off >= -256) {
        emit(j, 0xF8400000 | (off & 0x1ff) << 12 | H_ENV << 5 | rt);   /* ldur */
    } else {
        emit_movi(j, rt, (uint64_t)(int64_t)off);
        emit(j, 0xF8606800 | rt << 16 | H_ENV << 5 | rt);              /* ldr rt, [env, rt] */
    }
}

/*
 * To the guest pc in x16, registers live: env->pc set, then the vCPU's jump
 * cache probed inline (as gen_a64_lookup_and_goto_ptr, same TB flags as
 * this TB's: a branch changes none). An ishjit TB is entered at its fast
 * entry with the registers in place; a TCG TB after storing them; a miss
 * goes through helper_lookup_tb_ptr. Flags untouched until a store.
 */
static int lookup_shared = -1;
static void emit_lookup_body(IJ *j, bool shared);
static void emit_lookup(IJ *j)
{
    intptr_t o0 = ij_offset(j);

    if (lookup_shared) {
        /* the shared routine R_LOOKUP, x27 = this TB (its flags) */
        emit_checkenv(j, 3);
        emit_vstore(j, j->vwritten);
        emit_tb_addr(j, H_TMP2, 0);
        emit_jump_routine(j, ISHJIT_R_LOOKUP, H_TMP1);
    } else {
        emit_lookup_body(j, false);
    }
    ij_region(j, o0, 'L');
}

/*
 * The lookup (shared: x27 = the current TB, whose flags and cs_base the next
 * one must have; else this TB's, as constants)
 */
static void emit_lookup_body(IJ *j, bool shared)
{
    unsigned pb = j->page_bits;
    int jc_off = (int)offsetof(ArchCPU, parent_obj.tb_jmp_cache) -
                 (int)offsetof(ArchCPU, env);
    int bp_off = (int)offsetof(ArchCPU, parent_obj.breakpoints.tqh_first) -
                 (int)offsetof(ArchCPU, env);
    unsigned arr = offsetof(CPUJumpCache, array[0]);
    uint64_t f1 = shared ? 0 : (uint64_t)(tb_cflags(j->tb) & ~CF_INVALID) << 32 | j->tb->flags;
    uint32_t *miss[8], *to_tcg;
    int nm = 0;

    if (shared) {
        emit_env_far(j, false, H_TMP2, H_TMP1, routine_off(ISHJIT_R_SCRATCH));
    } else {
        emit_checkenv(j, 3);
        emit_vstore(j, j->vwritten);
    }
    QEMU_BUILD_BUG_ON(sizeof(((CPUJumpCache *)0)->array[0]) != 24);
    QEMU_BUILD_BUG_ON(offsetof(TranslationBlock, cflags) !=
                      offsetof(TranslationBlock, flags) + 4);
    QEMU_BUILD_BUG_ON(TB_JMP_PAGE_BITS != 8 || TB_JMP_GROUP_BITS != 8);

    emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
    /* tb_jmp_cache_hash_func(pc) * 24 in x27 */
    emit(j, 0xD340FC00 | pb << 16 | H_TMP0 << 5 | H_TMP1);             /* lsr x17, x16, #pb */
    emit(j, 0xCA400000 | H_TMP1 << 16 | 8 << 10 | H_TMP1 << 5 | H_TMP2);   /* eor x27, x17, x17, lsr #8 */
    emit(j, 0xCA400000 | H_TMP1 << 16 | 16 << 10 | H_TMP2 << 5 | H_TMP2);  /* eor x27, x27, x17, lsr #16 */
    emit(j, 0xD378DC00 | H_TMP2 << 5 | H_TMP2);                        /* lsl x27, x27, #8 */
    emit(j, 0x92781C00 | H_TMP2 << 5 | H_TMP2);                        /* and x27, x27, #0xff00 */
    emit(j, 0xD342FC00 | H_TMP0 << 5 | H_TMP1);                        /* lsr x17, x16, #2 */
    emit(j, 0xCA400000 | H_TMP0 << 16 | 10 << 10 | H_TMP1 << 5 | H_TMP1);  /* eor x17, x17, x16, lsr #10 */
    emit(j, 0xCA400000 | H_TMP0 << 16 | (pb + 8) << 10 | H_TMP1 << 5 | H_TMP1);
    emit(j, 0x92401C00 | H_TMP1 << 5 | H_TMP1);                        /* and x17, x17, #0xff */
    emit(j, 0xAA000000 | H_TMP1 << 16 | H_TMP2 << 5 | H_TMP2);         /* orr x27, x27, x17 */
    emit(j, 0x8B000400 | H_TMP2 << 16 | H_TMP2 << 5 | H_TMP2);         /* add x27, x27, x27, lsl #1 */
    emit(j, 0xD37DF000 | H_TMP2 << 5 | H_TMP2);                        /* lsl x27, x27, #3 */
    emit_ld_env_neg(j, H_TMP1, jc_off);                                 /* jc */
    emit(j, 0x8B000000 | H_TMP1 << 16 | H_TMP2 << 5 | H_TMP2);         /* e = jc + h */
    /* this half's stamp == entry's gen, entry's pc == pc, a TB */
    /* this half's stamp: stamp[(int64_t)pc < 0], as tb_jmp_cache_stamp */
    emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
    emit(j, 0xD37FFC00 | H_TMP0 << 5 | H_TMP0);                        /* lsr x16, x16, #63 */
    emit(j, 0x8B000C00 | H_TMP0 << 16 | H_TMP1 << 5 | H_TMP1);         /* add x17, x17, x16, lsl #3 */
    QEMU_BUILD_BUG_ON(offsetof(CPUJumpCache, stamp[1]) !=
                      offsetof(CPUJumpCache, stamp[0]) + 8);
    emit(j, i_ldr_x(H_TMP1, H_TMP1, offsetof(CPUJumpCache, stamp[0])));
    emit(j, i_ldr_x(H_TMP0, H_TMP2, arr + offsetof(typeof(((CPUJumpCache *)0)->array[0]), gen)));
    emit(j, 0xCA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);         /* eor x16, x16, x17 */
    miss[nm++] = j->ptr; emit(j, 0);
    emit(j, i_ldr_x(H_TMP0, H_TMP2, arr + offsetof(typeof(((CPUJumpCache *)0)->array[0]), pc)));
    emit(j, i_ldr_x(H_TMP1, H_ENV, offsetof(CPUARMState, pc)));
    emit(j, 0xCA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);
    miss[nm++] = j->ptr; emit(j, 0);
    emit(j, i_ldr_x(H_TMP2, H_TMP2, arr + offsetof(typeof(((CPUJumpCache *)0)->array[0]), tb)));
    miss[nm++] = j->ptr; emit(j, 0xB4000000 | H_TMP2);                 /* cbz x27, miss */
    /* its flags and cflags (no CF_INVALID), cs_base: this TB's */
    emit(j, i_ldr_x(H_TMP0, H_TMP2, offsetof(TranslationBlock, flags)));
    if (shared) {
        emit_env_far(j, true, H_TMP1, H_TMP1, routine_off(ISHJIT_R_SCRATCH));
        emit(j, i_ldr_x(H_TMP1, H_TMP1, offsetof(TranslationBlock, flags)));
    } else {
        emit_movi(j, H_TMP1, f1);
    }
    emit(j, 0xCA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);
    miss[nm++] = j->ptr; emit(j, 0);
    emit(j, i_ldr_x(H_TMP0, H_TMP2, offsetof(TranslationBlock, cs_base)));
    if (shared) {
        emit_env_far(j, true, H_TMP1, H_TMP1, routine_off(ISHJIT_R_SCRATCH));
        emit(j, i_ldr_x(H_TMP1, H_TMP1, offsetof(TranslationBlock, cs_base)));
    } else {
        emit_movi(j, H_TMP1, j->tb->cs_base);
    }
    emit(j, 0xCA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);
    miss[nm++] = j->ptr; emit(j, 0);
    /* no breakpoints */
    emit_ld_env_neg(j, H_TMP0, bp_off);
    miss[nm++] = j->ptr; emit(j, 0);
    /* an ishjit TB: its fast entry, registers in place */
    emit(j, 0x79400000 | (offsetof(TranslationBlock, ishjit_fast) / 2) << 10 |
            H_TMP2 << 5 | H_TMP0);                                      /* ldrh w16, [x27, #fast] */
    to_tcg = j->ptr; emit(j, 0);                                        /* cbz w16, tcg */
    emit(j, i_ldr_x(H_TMP1, H_TMP2, offsetof(TranslationBlock, tc.ptr)));
    emit(j, 0x8B000000 | H_TMP0 << 16 | H_TMP1 << 5 | H_TMP1);         /* add x17, x17, x16 */
    emit(j, i_br(H_TMP1));
    /* a TCG TB: registers stored (its code pointer kept across the routine) */
    *to_tcg = 0x34000000 | ((j->ptr - to_tcg) & 0x7ffff) << 5 | H_TMP0;
    emit(j, i_ldr_x(H_TMP0, H_TMP2, offsetof(TranslationBlock, tc.ptr)));
    emit_env_far(j, false, H_TMP0, H_TMP1, routine_off(ISHJIT_R_SCRATCH));
    emit_sync(j);
    emit_env_far(j, true, H_TMP0, H_TMP0, routine_off(ISHJIT_R_SCRATCH));
    emit(j, i_br(H_TMP0));
    /* a miss: helper_lookup_tb_ptr */
    for (int k = 0; k < nm; k++) {
        uint32_t base = miss[k] == miss[2] ? (*miss[k] & 0xFF00001F) :
                        0xB5000000 | H_TMP0;                            /* cbnz x16 */
        *miss[k] = base | ((j->ptr - miss[k]) & 0x7ffff) << 5;
    }
    emit_sync(j);
    emit(j, i_mov_x(0, H_ENV));
    emit_movi(j, H_TMP0, (uintptr_t)helper_lookup_tb_ptr);
    emit(j, i_blr(H_TMP0));
    emit(j, i_br(0));
}

/* To @dest: pc updated, then chained, or looked up */
static void emit_jump(IJ *j, int which, vaddr dest)
{
    if (((dest ^ j->pc_first) & TARGET_PAGE_MASK) == 0) {
        intptr_t o0 = ij_offset(j);

        emit_vstore(j, j->vwritten);
        emit_goto_tb(j, which, dest);
        ij_region(j, o0, 'J');
    } else {
        emit_pc_rel(j, H_TMP0, dest);
        emit_lookup(j);
        return;
        emit(j, i_mov_x(0, H_ENV));
        emit_movi(j, H_TMP0, (uintptr_t)helper_lookup_tb_ptr);
        emit(j, i_blr(H_TMP0));
        emit(j, i_br(0));
    }
}

/* The guest value of a register read by a branch (host reg or scratch) */
static int branch_src(IJ *j, int g)
{
    if (g == 31) {
        return 31;                          /* xzr */
    }
    if (g2h[g] != H_NONE) {
        return g2h[g];
    }
    emit(j, i_ldr_x(H_TMP1, H_ENV, xreg_off(g)));
    return H_TMP1;
}


/* A guest integer load or store (one register or a pair) */
enum { M_IMM, M_REG, M_LIT };
/*
 * Ordered, exclusive and atomic accesses: LDAR/LDAPR/STLR as host
 * acquire/release accesses; LSE atomics (LD<op>, SWP, CAS) run on the host
 * address; LDXR/STXR follow QEMU's model (the exclusive address and value in
 * env, STXR a compare-and-swap with them), which holds with other vCPUs
 * and with anything ishjit emits between the two.
 */
enum { A_NONE, A_LDAR, A_LDAPR, A_STLR, A_LSE, A_CAS, A_LDX, A_STX };
enum { WB_NONE, WB_PRE, WB_POST };

typedef struct {
    uint8_t size;           /* log2 bytes per register */
    uint8_t opc;            /* bits [23:22] of the single-register forms */
    bool load, pair, sign, dest32;
    bool vec;               /* SIMD&FP register (size up to 4: Q) */
    uint8_t akind;          /* A_*: ordered, exclusive or atomic access */
    uint8_t rs;             /* (atomics, cas: Rs; stxr: the status) */
    bool unpriv;            /* ldtr/sttr: through the user MMU index */
    int8_t midx;            /* the core MMU index of the access */
    uint8_t rt, rt2, rn, rm;
    uint8_t mode, wb, option, shift;
    bool prefetch;
    int64_t imm;            /* byte offset (or the literal's pc offset) */
} MemInfo;

static bool decode_atomic(uint32_t insn, MemInfo *m)
{
    unsigned size = insn >> 30;

    m->mode = M_IMM;
    m->size = size;
    m->rs = (insn >> 16) & 31;
    if ((insn & 0x3FFFFC00) == (0x08DFFC00 & 0x3FFFFC00)) {       /* ldar */
        m->akind = A_LDAR;
        m->load = true;
        m->opc = 1;
        return true;
    }
    if ((insn & 0x3FFFFC00) == (0x089FFC00 & 0x3FFFFC00)) {       /* stlr */
        m->akind = A_STLR;
        return true;
    }
    if ((insn & 0x3FFFFC00) == 0x38BFC000) {                     /* ldapr */
        m->akind = A_LDAPR;
        m->load = true;
        m->opc = 1;
        return true;
    }
    if ((insn & 0x3FFF7C00) == 0x085F7C00) {                     /* ldxr, ldaxr */
        m->akind = A_LDX;
        m->load = true;
        return true;
    }
    if ((insn & 0x3FE07C00) == 0x08007C00) {                     /* stxr, stlxr */
        /* (status == data or base: UNPREDICTABLE, left to TCG) */
        if (m->rs == m->rt || (m->rs == m->rn && m->rn != 31)) {
            return false;
        }
        m->akind = A_STX;
        return true;
    }
    if ((insn & 0x3FA07C00) == 0x08A07C00) {                     /* cas(a)(l) */
        m->akind = A_CAS;
        return true;
    }
    if ((insn & 0x3B200C00) == 0x38200000 && !(insn & (1u << 26))) {
        unsigned o3 = (insn >> 15) & 1, opc = (insn >> 12) & 7;

        if (o3 == 0 || opc == 0) {                               /* ld<op>, swp */
            m->akind = A_LSE;
            return true;
        }
    }
    return false;
}

static bool decode_mem(uint32_t insn, MemInfo *m, bool fp_ok)
{
    memset(m, 0, sizeof(*m));
    m->rt = insn & 31;
    m->rn = (insn >> 5) & 31;
    if (decode_atomic(insn, m)) {
        return true;
    }
    static int nosp = -1;

    if (nosp < 0) {
        nosp = getenv("ISHJIT_NOSP") != NULL;
    }
    if (m->rn == 31 && nosp &&
        (insn & 0x3B000000) != 0x18000000) {
        return false;
    }

    if ((insn & 0x3B000000) == 0x39000000 ||        /* unsigned immediate */
        (insn & 0x3B200000) == 0x38000000 ||        /* imm9 forms */
        (insn & 0x3B200C00) == 0x38200800) {        /* register offset */
        unsigned size = insn >> 30, opc = (insn >> 22) & 3;

        if (insn & (1u << 26)) {
            /* SIMD&FP: B/H/S/D/Q; opc bit 1 means Q (with size 0) */
            if (!fp_ok || ((opc & 2) && size != 0)) {
                return false;
            }
            m->vec = true;
            if (opc & 2) {
                size = 4;
            }
            opc &= 1;
        }
        if ((insn & 0x3B000000) == 0x39000000) {
            m->mode = M_IMM;
            m->imm = (int64_t)((insn >> 10) & 0xfff) << size;
        } else if ((insn & 0x3B200000) == 0x38000000) {
            unsigned kind = (insn >> 10) & 3;

            m->unpriv = kind == 2;                  /* ldtr/sttr: no writeback */
            m->mode = M_IMM;
            m->imm = sextract32(insn, 12, 9);
            m->wb = kind == 1 ? WB_POST : kind == 3 ? WB_PRE : WB_NONE;
        } else {
            unsigned option = (insn >> 13) & 7;

            if (!(option & 2)) {
                return false;
            }
            m->mode = M_REG;
            m->rm = (insn >> 16) & 31;
            m->option = option;
            m->shift = (insn & (1u << 12)) ? size : 0;
        }
        if (m->vec) {
            m->size = size;
            m->load = opc;
            return true;
        }
        if (opc == 2 && size == 3) {
            if (m->wb != WB_NONE) {
                return false;
            }
            m->prefetch = true;                     /* prfm, prfum */
            return true;
        }
        if ((opc == 3 && size >= 2) || (opc == 2 && size == 3)) {
            return false;
        }
        m->size = size;
        m->opc = opc;
        m->load = opc != 0;
        m->sign = opc >= 2;
        m->dest32 = opc == 3;
        return true;
    }
    if ((insn & 0x3A000000) == 0x28000000) {        /* pairs */
        unsigned opc = insn >> 30, mode = (insn >> 23) & 3;
        bool load = insn & (1u << 22);

        if (insn & (1u << 26)) {
            if (!fp_ok || opc == 3) {
                return false;
            }
            m->vec = true;                          /* S, D, Q pairs */
            m->pair = true;
            m->load = load;
            m->size = 2 + opc;
            m->opc = opc;
            m->rt2 = (insn >> 10) & 31;
            m->mode = M_IMM;
            m->imm = (int64_t)sextract32(insn, 15, 7) << m->size;
            m->wb = mode == 1 ? WB_POST : mode == 3 ? WB_PRE : WB_NONE;
            return true;
        }
        if (opc == 3 || (opc == 1 && (!load || mode == 0))) {
            return false;                           /* (stgp, ldpsw nt...) */
        }
        m->pair = true;
        m->load = load;
        m->size = opc == 2 ? 3 : 2;
        m->sign = opc == 1;
        m->opc = opc;
        m->rt2 = (insn >> 10) & 31;
        m->mode = M_IMM;
        m->imm = (int64_t)sextract32(insn, 15, 7) << m->size;
        m->wb = mode == 1 ? WB_POST : mode == 3 ? WB_PRE : WB_NONE;
        return true;
    }
    if ((insn & 0x3B000000) == 0x18000000) {        /* load literal */
        unsigned opc = insn >> 30;

        if (insn & (1u << 26)) {
            if (!fp_ok || opc == 3) {
                return false;
            }
            m->vec = true;                          /* S, D, Q */
            m->mode = M_LIT;
            m->imm = (int64_t)sextract32(insn, 5, 19) << 2;
            m->load = true;
            m->size = 2 + opc;
            return true;
        }
        m->mode = M_LIT;
        m->imm = (int64_t)sextract32(insn, 5, 19) << 2;
        if (opc == 3) {
            m->prefetch = true;
            return true;
        }
        m->load = true;
        m->size = opc == 1 ? 3 : 2;
        m->sign = opc == 2;
        m->opc = opc == 2 ? 2 : 1;
        return true;
    }
    return false;
}

/* guest registers a memory access reads / writes (31 = SP for the base only) */
static void mem_regs(const MemInfo *m, uint32_t *reads, uint32_t *writes)
{
    *reads = *writes = 0;
    if (m->prefetch) {
        return;
    }
    if (m->mode != M_LIT) {
        *reads |= 1u << m->rn;
        if (m->wb != WB_NONE) {
            *writes |= 1u << m->rn;
        }
    }
    if (m->mode == M_REG && m->rm != 31) {
        *reads |= 1u << m->rm;
    }
    if (m->vec) {
        return;                                     /* (V registers apart) */
    }
    switch (m->akind) {
    case A_LSE:                                     /* Rs in, Rt old value */
        if (m->rs != 31) {
            *reads |= 1u << m->rs;
        }
        if (m->rt != 31) {
            *writes |= 1u << m->rt;
        }
        return;
    case A_CAS:                                     /* Rs expected/old, Rt new */
        if (m->rs != 31) {
            *reads |= 1u << m->rs;
            *writes |= 1u << m->rs;
        }
        if (m->rt != 31) {
            *reads |= 1u << m->rt;
        }
        return;
    case A_STX:                                     /* Ws status, Xt data */
        if (m->rs != 31) {
            *writes |= 1u << m->rs;
        }
        if (m->rt != 31) {
            *reads |= 1u << m->rt;
        }
        return;
    }
    if (m->rt != 31) {
        *(m->load ? writes : reads) |= 1u << m->rt;
    }
    if (m->pair && m->rt2 != 31) {
        *(m->load ? writes : reads) |= 1u << m->rt2;
    }
}



/*
 * Slow path of LSE atomics, CAS, LDXR and STXR (the TLB missed, the address
 * is misaligned or not plain RAM): QEMU's own operations, on the guest
 * registers in env (the block's save helper stored them).
 */
#define ISHJIT_ATOMIC(op, sz, a, v, oi, ra)                                  \
    ((sz) == 0 ? cpu_atomic_##op##b_mmu(env, a, v, oi, ra) :                \
     (sz) == 1 ? cpu_atomic_##op##w_le_mmu(env, a, v, oi, ra) :             \
     (sz) == 2 ? cpu_atomic_##op##l_le_mmu(env, a, v, oi, ra) :             \
                 cpu_atomic_##op##q_le_mmu(env, a, v, oi, ra))

static uint64_t ishjit_cmpxchg(CPUARMState *env, int sz, uint64_t a,
                               uint64_t c, uint64_t v, MemOpIdx oi, uintptr_t ra)
{
    switch (sz) {
    case 0:
        return cpu_atomic_cmpxchgb_mmu(env, a, c, v, oi, ra);
    case 1:
        return cpu_atomic_cmpxchgw_le_mmu(env, a, c, v, oi, ra);
    case 2:
        return cpu_atomic_cmpxchgl_le_mmu(env, a, c, v, oi, ra);
    default:
        return cpu_atomic_cmpxchgq_le_mmu(env, a, c, v, oi, ra);
    }
}

void ishjit_atomic_slow(CPUARMState *env, uint64_t addr, uint32_t insn,
                        uintptr_t ra, int mmu_idx);
void ishjit_atomic_slow(CPUARMState *env, uint64_t addr, uint32_t insn,
                        uintptr_t ra, int mmu_idx)
{
    MemInfo m;
    MemOpIdx oi;
    uint64_t mask, rs, rt, old;

    decode_mem(insn, &m, false);
    mask = m.size == 3 ? ~0ull : (1ull << (8 << m.size)) - 1;
    oi = make_memop_idx(m.size | MO_LE | MO_ALIGN, mmu_idx);
    rs = m.rs == 31 ? 0 : env->xregs[m.rs] & mask;
    rt = m.rt == 31 ? 0 : env->xregs[m.rt] & mask;

    switch (m.akind) {
    case A_LSE:
        if ((insn >> 15) & 1) {
            old = ISHJIT_ATOMIC(xchg, m.size, addr, rs, oi, ra);
        } else {
            switch ((insn >> 12) & 7) {
            case 0: old = ISHJIT_ATOMIC(fetch_add, m.size, addr, rs, oi, ra); break;
            case 1: old = ISHJIT_ATOMIC(fetch_and, m.size, addr, ~rs & mask, oi, ra); break;
            case 2: old = ISHJIT_ATOMIC(fetch_xor, m.size, addr, rs, oi, ra); break;
            case 3: old = ISHJIT_ATOMIC(fetch_or, m.size, addr, rs, oi, ra); break;
            case 4: old = ISHJIT_ATOMIC(fetch_smax, m.size, addr, rs, oi, ra); break;
            case 5: old = ISHJIT_ATOMIC(fetch_smin, m.size, addr, rs, oi, ra); break;
            case 6: old = ISHJIT_ATOMIC(fetch_umax, m.size, addr, rs, oi, ra); break;
            default: old = ISHJIT_ATOMIC(fetch_umin, m.size, addr, rs, oi, ra); break;
            }
        }
        if (m.rt != 31) {
            env->xregs[m.rt] = old & mask;
        }
        break;
    case A_CAS:
        old = ishjit_cmpxchg(env, m.size, addr, rs, rt, oi, ra);
        if (m.rs != 31) {
            env->xregs[m.rs] = old & mask;
        }
        break;
    case A_LDX:
        switch (m.size) {
        case 0: old = cpu_ldb_mmu(env, addr, oi, ra); break;
        case 1: old = cpu_ldw_mmu(env, addr, oi, ra); break;
        case 2: old = cpu_ldl_mmu(env, addr, oi, ra); break;
        default: old = cpu_ldq_mmu(env, addr, oi, ra); break;
        }
        env->exclusive_addr = addr;
        env->exclusive_val = old;
        if (m.rt != 31) {
            env->xregs[m.rt] = old;
        }
        break;
    case A_STX: {
        uint64_t status = 1;

        if (addr == env->exclusive_addr) {
            old = ishjit_cmpxchg(env, m.size, addr, env->exclusive_val, rt, oi, ra);
            status = (old & mask) != (env->exclusive_val & mask);
        }
        env->exclusive_addr = -1;
        if (m.rs != 31) {
            env->xregs[m.rs] = status;
        }
        break;
    }
    }
}

enum { K_STOP, K_DP, K_NOP, K_ADR, K_MEM, K_FP, K_COPY, K_CLREX, K_MRS, K_MSR,
       K_MRSCONST, K_DAIFSET, K_DAIFCLR,
       K_B, K_BL, K_BCOND, K_CB, K_TB, K_BREG, K_SVC, K_BRK, K_ERET, K_ISB };

/* System instructions ishjit handles itself (others: TCG) */
static int classify_sys(uint32_t insn, IJ *j, unsigned *off, bool *spcheck)
{
    unsigned crm = (insn >> 8) & 15;

    if ((insn & 0xFFFFF01F) == 0xD503201F) {            /* hints */
        unsigned h = (insn >> 5) & 0x7f;

        /*
         * Architectural no-ops here (PAC and BTI hints too: no FEAT_PAuth,
         * FEAT_BTI with a13); WFE, WFI, ESB and the rest go to TCG.
         */
        if (h == 0 || h == 1 || h == 4 || h == 5 || h == 7 ||
            (h >= 8 && h <= 15) || h == 17 || h == 18 || h == 20 ||
            h == 22 || (h >= 24 && h <= 31) || h == 32 || h == 34 ||
            h == 36 || h == 38) {
            return K_NOP;
        }
        return K_STOP;
    }
    if ((insn & 0xFFFFF0FF) == 0xD50330BF ||            /* dmb */
        (insn & 0xFFFFF0FF) == 0xD503309F) {            /* dsb */
        if ((insn & 0xFFFFF0FF) == 0xD503309F && (crm == 0 || crm == 4)) {
            return K_NOP;                               /* ssbb, pssbb */
        }
        return K_COPY;
    }
    if ((insn & 0xFFFFF0FF) == 0xD50330DF) {            /* isb */
        return K_ISB;
    }
    if ((insn & 0xFFFFF0FF) == 0xD503305F) {            /* clrex */
        return K_CLREX;
    }
    /* interrupt masking (the kernel's local_irq_*): at EL1, no traps */
    if ((insn & 0xFFFFF0FF) == 0xD50340DF) {            /* msr daifset, #imm */
        return j->el == 1 ? K_DAIFSET : K_STOP;
    }
    if ((insn & 0xFFFFF0FF) == 0xD50340FF) {            /* msr daifclr, #imm */
        return j->el == 1 ? K_DAIFCLR : K_STOP;
    }
    if ((insn & 0xFFE0001F) == 0xD4000001) {            /* svc */
        return j->el == 0 && !j->fgt_svc ? K_SVC : K_STOP;
    }
    if ((insn & 0xFFE0001F) == 0xD4200000) {            /* brk */
        return K_BRK;
    }
    if (insn == 0xD69F03E0) {                           /* eret */
        return j->el == 1 && !j->trap_eret ? K_ERET : K_STOP;
    }
    /*
     * Other system registers, from QEMU's own table (as TCG's handle_sys):
     * plain fields read, or written where TCG would not end the TB (no TB
     * flags depend on them); constants; no-op SYS instructions (cache
     * maintenance QEMU ignores). Only with an access check known to pass.
     */
    if ((insn & 0xFFC00000) == 0xD5000000 && ((insn >> 19) & 3)) {
        bool isread = insn & (1u << 21);
        int op0 = (insn >> 19) & 3, op1 = (insn >> 16) & 7, crn = (insn >> 12) & 15;
        int op2 = (insn >> 5) & 7;
        uint32_t key = ENCODE_AA64_CP_REG(CP_REG_ARM64_SYSREG_CP, crn, crm, op0, op1, op2);
        const ARMCPRegInfo *ri = get_arm_cp_reginfo(env_archcpu(j->env)->cp_regs, key);
        int t;

        if (!ri || !cp_access_ok(j->el, ri, isread) ||
            (ri->fgt && j->fgt_active)) {
            goto other;
        }
        if (ri->accessfn &&
            arm_cpreg_ishplus_access(ri, j->el, j->el2_el3) != 1) {
            goto other;
        }
        t = ri->type;
        if (op0 == 1 && (t & ARM_CP_SPECIAL_MASK) != ARM_CP_NOP) {
            goto other;                 /* (SYS: only what QEMU ignores) */
        }
        if (t & ARM_CP_SPECIAL_MASK) {
            return (t & ARM_CP_SPECIAL_MASK) == ARM_CP_NOP ? K_NOP : K_STOP;
        }
        if (t & (ARM_CP_IO | ARM_CP_RAISES_EXC | ARM_CP_NEWEL | ARM_CP_FPU |
                 ARM_CP_SVE | ARM_CP_SME)) {
            goto other;
        }
        if (t & ARM_CP_CONST) {
            if (!isread) {
                return K_NOP;
            }
            *off = 0;
            j->sysconst = ri->resetvalue;
            return K_MRSCONST;
        }
        if (!ri->fieldoffset) {
            goto other;
        }
        if (ri->fieldoffset % 8 || ri->fieldoffset / 8 > 4095) {
            goto other;                 /* (beyond a plain ldr/str) */
        }
        if (isread && !ri->readfn) {
            *off = ri->fieldoffset;
            return K_MRS;
        }
        if (!isread && !ri->writefn && (t & ARM_CP_SUPPRESS_TB_END)) {
            *off = ri->fieldoffset;
            return K_MSR;
        }
    }
other:
    /* thread pointers (plain fields, no traps without fine-grained traps) */
    if (!j->fgt_active) {
        switch (insn & 0xFFFFFFE0) {
        case 0xD53BD040:                                /* mrs tpidr_el0 */
            *off = offsetof(CPUARMState, cp15.tpidr_el[0]);
            return K_MRS;
        case 0xD53BD060:                                /* mrs tpidrro_el0 */
            *off = offsetof(CPUARMState, cp15.tpidrro_el[0]);
            return K_MRS;
        case 0xD51BD040:                                /* msr tpidr_el0 */
            *off = offsetof(CPUARMState, cp15.tpidr_el[0]);
            return K_MSR;
        case 0xD538D080:                                /* mrs tpidr_el1 */
            *off = offsetof(CPUARMState, cp15.tpidr_el[1]);
            return j->el == 1 ? K_MRS : K_STOP;
        case 0xD5384100:                                /* mrs sp_el0 */
        case 0xD5184100:                                /* msr sp_el0 */
            /* (Linux's current task; accessible at EL1 unless PSTATE.SP 0) */
            *off = offsetof(CPUARMState, sp_el[0]);
            *spcheck = true;
            if (j->el != 1) {
                return K_STOP;
            }
            return (insn & 0x00200000) ? K_MRS : K_MSR;
        }
    }
    return K_STOP;
}

/*
 * An FP/AdvSIMD data-processing instruction ishjit copies (ishjit-simd.c
 * checks it is one the A13 implements, against LLVM, exhaustively): V
 * registers are the host's own; GPR fields are remapped. Left to TCG: FP
 * arithmetic when the FPCR is not all default (the host's is), and
 * saturating ops (FPSR.QC is not folded back from the host yet).
 */
static bool fp_copyable(uint32_t insn, SimdOperands *g, IJ *j)
{
    if (!j->fp_ok || (insn & 0x0E000000) != 0x0E000000 ||
        !ishjit_simd_decode(insn, g)) {
        return false;
    }
    if (g->sets_qc || (g->reads_fpcr_mode && !j->fp_default)) {
        return false;
    }
    return true;
}

/* GPRs read / written, V registers touched (conservatively: every V field) */
static void fp_regs(uint32_t insn, const SimdOperands *g, uint32_t *reads,
                    uint32_t *writes, uint32_t *vregs)
{
    int f[4] = { insn & 31, (insn >> 5) & 31, (insn >> 16) & 31, (insn >> 10) & 31 };
    uint8_t role[4] = { g->rd, g->rn, g->rm, SIMD_NOT_GPR };

    *reads = *writes = *vregs = 0;
    for (int k = 0; k < 4; k++) {
        if (role[k] == SIMD_NOT_GPR) {
            *vregs |= 1u << f[k];
        } else if (f[k] != 31) {
            if (k == 0 && g->rd_written) {
                *writes |= 1u << f[k];
            } else {
                *reads |= 1u << f[k];
            }
        }
    }
    /* (by element on 16-bit elements: Vm is 4 bits, bit 20 is the index) */
    if (role[2] == SIMD_NOT_GPR) {
        *vregs |= 1u << (f[2] & 15);
    }
    /* tbl/tbx: Vn..Vn+len (mod 32) */
    if ((insn & 0xBF208C00) == 0x0E000000) {
        int len = (insn >> 13) & 3;

        for (int k = 0; k <= len; k++) {
            *vregs |= 1u << ((f[1] + k) & 31);
        }
    }
}

static int classify(uint32_t insn, OperandInfo *o, MemInfo *m, bool mem_ok,
                    bool fp_ok, SimdOperands *g, IJ *j)
{
    if (mem_ok && decode_mem(insn, m, fp_ok)) {
        m->midx = j->mmu_idx;
        if (m->unpriv && j->unpriv) {
            /* as TCG's get_a64_user_mem_index */
            ARMMMUIdx cur = core_to_aa64_mmu_idx(j->mmu_idx), user = cur;

            switch (cur) {
            case ARMMMUIdx_E10_1:
            case ARMMMUIdx_E10_1_PAN:
                user = ARMMMUIdx_E10_0;
                break;
            case ARMMMUIdx_E20_2:
            case ARMMMUIdx_E20_2_PAN:
                user = ARMMMUIdx_E20_0;
                break;
            default:
                break;
            }
            m->midx = arm_to_core_mmu_idx(user);
            if (tlb_fast_off(m->midx) < -512) {
                return K_STOP;
            }
        }
        return m->prefetch ? K_NOP : K_MEM;
    }
    if ((insn & 0xFF000000) == 0xD5000000 || (insn & 0xFF000000) == 0xD4000000 ||
        insn == 0xD69F03E0) {
        unsigned off = 0;
        bool spcheck = false;
        int k = classify_sys(insn, j, &off, &spcheck);

        o->sysoff = off;
        o->spcheck = spcheck;
        o->sysconst = j->sysconst;
        return k;
    }
    if (fp_copyable(insn, g, j)) {
        return K_FP;
    }

    if (insn == 0xD503201F) {
        return K_NOP;
    }
    if ((insn & 0x1F000000) == 0x10000000) {
        return K_ADR;
    }
    if ((insn & 0xFC000000) == 0x14000000) {
        return K_B;
    }
    if ((insn & 0xFC000000) == 0x94000000) {
        return K_BL;
    }
    if ((insn & 0xFF000010) == 0x54000000) {
        return K_BCOND;
    }
    if ((insn & 0x7E000000) == 0x34000000) {
        return K_CB;
    }
    if ((insn & 0x7E000000) == 0x36000000) {
        return K_TB;
    }
    /* br, blr, ret (no pointer authentication) */
    if ((insn & 0xFF9FFC1F) == 0xD61F0000 && ((insn >> 21) & 3) != 3) {
        return K_BREG;
    }
    if (decode_dp(insn, o)) {
        return K_DP;
    }
    return K_STOP;
}


/* A slow path to emit after the block: where its fast path branches from */
typedef struct {
    uint32_t *cbnz;         /* the fast path's cbnz x17, (stub) */
    uint32_t *resume;       /* after the access (before any writeback) */
    MemInfo m;
    uint32_t insn;          /* (atomics: the guest insn, for the C helper) */
    bool flags;             /* NZCV valid in the host at this insn */
    bool allslow;           /* (debugging: always the slow path) */
    uint32_t *cbnz2;        /* (atomics: the alignment check's cbnz x17) */
    int gkind;              /* (shared checks: 1 first, 2 later; out of line: the softmmu way) */
    uint32_t *gbr;          /* (its branch there) */
    int gslot;
    vaddr ipc;
} IJSlow;

/* x27 = host address of the last byte, x16 = guest address of the last byte */
static void emit_atomic_access(IJ *j, const MemInfo *m, int accsize)
{
    int sz = m->size;
    int rt = m->rt == 31 ? 31 : g2h[m->rt];
    int rs = m->rs == 31 ? 31 : g2h[m->rs];
    uint32_t *fail, *done;

    if (accsize > 1) {
        emit_addi(j, H_TMP2, H_TMP2, -(accsize - 1));       /* exact address */
        emit_addi(j, H_TMP0, H_TMP0, -(accsize - 1));       /* (guest: first byte) */
    }
    switch (m->akind) {
    case A_LDAR:
    case A_LDAPR:
    case A_STLR: {
        uint32_t op = m->akind == A_LDAR ? 0x08DFFC00 :
                      m->akind == A_LDAPR ? 0x38BFC000 : 0x089FFC00;
        bool envr = rt == H_NONE;

        if (envr) {
            rt = H_TMP1;
            if (m->akind == A_STLR) {
                emit(j, i_ldr_x(rt, H_ENV, xreg_off(m->rt)));
            }
        }
        emit(j, (uint32_t)sz << 30 | op | H_TMP2 << 5 | rt);
        if (envr && m->akind != A_STLR) {
            emit(j, i_str_x(rt, H_ENV, xreg_off(m->rt)));
        }
        break;
    }
    case A_LSE:
    case A_CAS: {
        uint32_t insn;
        int ts = rs, tt = rt;

        if (rs == H_NONE) {
            ts = H_TMP0;
            emit(j, i_ldr_x(ts, H_ENV, xreg_off(m->rs)));
        }
        if (rt == H_NONE) {
            tt = H_TMP1;
            if (m->rt == m->rs) {
                tt = ts;
            } else if (m->akind == A_CAS) {
                emit(j, i_ldr_x(tt, H_ENV, xreg_off(m->rt)));
            }
        }
        /* the guest's own instruction (with its ordering), on host registers */
        insn = j->cur_insn & ~(0x1Fu << 16 | 0x1Fu << 5 | 0x1Fu);
        emit(j, insn | ts << 16 | H_TMP2 << 5 | tt);
        if (m->akind == A_LSE && rt == H_NONE && m->rt != 31) {
            emit(j, i_str_x(tt, H_ENV, xreg_off(m->rt)));
        }
        if (m->akind == A_CAS && rs == H_NONE) {
            emit(j, i_str_x(ts, H_ENV, xreg_off(m->rs)));
        }
        break;
    }
    case A_LDX: {
        /* the value (zero-extended) in x17 */
        if (j->cur_insn & (1u << 15)) {
            emit(j, (uint32_t)sz << 30 | 0x08DFFC00 | H_TMP2 << 5 | H_TMP1);   /* ldar */
        } else {
            emit(j, (uint32_t)sz << 30 | 0x38400000 | H_TMP2 << 5 | H_TMP1);   /* ldur */
        }
        emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, exclusive_addr)));
        emit(j, i_str_x(H_TMP1, H_ENV, offsetof(CPUARMState, exclusive_val)));
        if (m->rt != 31) {
            if (rt == H_NONE) {
                emit(j, i_str_x(H_TMP1, H_ENV, xreg_off(m->rt)));
            } else {
                emit(j, i_mov_x(rt, H_TMP1));
            }
        }
        break;
    }
    case A_STX: {
        /* the address must be the exclusive one; then cas against its value */
        emit(j, i_ldr_x(H_TMP1, H_ENV, offsetof(CPUARMState, exclusive_addr)));
        emit(j, 0xCA000000 | H_TMP0 << 16 | H_TMP1 << 5 | H_TMP1);   /* eor x17, x17, x16 */
        fail = j->ptr;
        emit(j, 0);                                                  /* cbnz x17, fail */
        emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, exclusive_val)));
        emit(j, i_mov_x(H_TMP1, H_TMP0));
        emit(j, (uint32_t)sz << 30 | 0x08E0FC00 | H_TMP0 << 16 | H_TMP2 << 5 | rt);  /* casal x16, xt, [x27] */
        emit(j, 0xCA000000 | H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);   /* eor x16, x16, x17 */
        if (sz < 3) {
            /* (casal of a byte/half/word: compare those bits) */
            emit(j, 0x92400000 | ((8 << sz) - 1) << 10 | H_TMP0 << 5 | H_TMP0);
        }
        emit(j, 0xDAC01000 | H_TMP0 << 5 | H_TMP0);                  /* clz x16, x16 */
        emit(j, 0xD346FC00 | H_TMP0 << 5 | H_TMP0);                  /* lsr x16, x16, #6 */
        emit(j, 0xD2400000 | H_TMP0 << 5 | H_TMP0);                  /* eor x16, x16, #1 */
        done = j->ptr;
        emit(j, 0);                                                  /* b done */
        *fail = 0xB5000000 | ((j->ptr - fail) & 0x7ffff) << 5 | H_TMP1;
        emit(j, 0xD2800020 | H_TMP0);                                /* mov x16, #1 */
        *done = i_b(j->ptr - done);
        emit(j, 0x92800000 | H_TMP1);                                /* mov x17, #-1 */
        emit(j, i_str_x(H_TMP1, H_ENV, offsetof(CPUARMState, exclusive_addr)));
        if (m->rs != 31) {
            if (rs == H_NONE) {
                emit(j, i_str_x(H_TMP0, H_ENV, xreg_off(m->rs)));
            } else {
                emit(j, i_mov_x(rs, H_TMP0));
            }
        }
        break;
    }
    }
}

/*
 * A guest integer load/store: the address in x16 (+ size - 1, so that the
 * TLB check also catches an access crossing a page), an inline check of
 * QEMU's softmmu TLB that leaves the flags alone, then the access itself on
 * the host address. Misses go to a slow path (after the block).
 */
/* The access's first byte to x16 (from its base and offset registers) */
static void emit_addr(IJ *j, const MemInfo *m, vaddr ipc)
{
    if (m->mode == M_LIT) {
        emit_pc_rel(j, H_TMP0, ipc + m->imm);
    } else {
        int base = g2h[m->rn];

        if (base == H_NONE) {
            emit(j, i_ldr_x(H_TMP0, H_ENV, xreg_off(m->rn)));
            base = H_TMP0;
        }
        if (m->mode == M_REG) {
            int rm = m->rm == 31 ? 31 : g2h[m->rm];

            if (rm == H_NONE) {
                emit(j, i_ldr_x(H_TMP1, H_ENV, xreg_off(m->rm)));
                rm = H_TMP1;
            }
            /* add x16, base, rm, <extend> #shift */
            emit(j, 0x8B200000 | rm << 16 | m->option << 13 | m->shift << 10 |
                    base << 5 | H_TMP0);
        } else {
            emit_addi(j, H_TMP0, base, m->wb == WB_POST ? 0 : m->imm);
        }
        /*
         * (no top-byte-ignore here: a tagged address cannot match a TLB
         * entry, the slow path removes the tag; the index ignores it)
         */
    }
}

/* The access itself, x27 the host address of its last byte */
static void emit_access(IJ *j, const MemInfo *m, int accsize)
{
    if (m->akind != A_NONE) {
        emit_atomic_access(j, m, accsize);
    } else if (m->vec && !m->pair) {
        unsigned sz = m->size == 4 ? 0 : m->size;
        unsigned opcv = (m->size == 4 ? 2 : 0) | m->load;

        /* ldur/stur Vt, [x27, #-(size - 1)] */
        emit(j, sz << 30 | 0x3C000000 | opcv << 22 |
                ((-(accsize - 1)) & 0x1ff) << 12 | H_TMP2 << 5 | m->rt);
    } else if (m->vec) {
        emit_addi(j, H_TMP2, H_TMP2, -(accsize - 1));
        /* ldp/stp St/Dt/Qt, [x27] */
        emit(j, (m->size - 2) << 30 | 0x2D000000 | (m->load ? 1 << 22 : 0) |
                m->rt2 << 10 | H_TMP2 << 5 | m->rt);
    } else if (!m->pair) {
        int rt = m->rt == 31 ? 31 : g2h[m->rt];
        bool in_env = rt == H_NONE;

        if (in_env) {
            rt = H_TMP1;
            if (!m->load) {
                emit(j, i_ldr_x(rt, H_ENV, xreg_off(m->rt)));
            }
        }
        /* ldur/stur (any size, sign) rt, [x27, #-(size - 1)] */
        emit(j, m->size << 30 | 0x38000000 | m->opc << 22 |
                ((-(accsize - 1)) & 0x1ff) << 12 | H_TMP2 << 5 | rt);
        if (in_env && m->load) {
            emit(j, i_str_x(rt, H_ENV, xreg_off(m->rt)));
        }
    } else {
        int rt = m->rt == 31 ? 31 : g2h[m->rt];
        int rt2 = m->rt2 == 31 ? 31 : g2h[m->rt2];
        bool e1 = rt == H_NONE, e2 = rt2 == H_NONE;

        emit_addi(j, H_TMP2, H_TMP2, -(accsize - 1));
        if (e1) {
            rt = H_TMP0;
        }
        if (e2) {
            rt2 = H_TMP1;
        }
        if (!m->load) {
            if (e1) {
                emit(j, i_ldr_x(rt, H_ENV, xreg_off(m->rt)));
            }
            if (e2) {
                emit(j, i_ldr_x(rt2, H_ENV, xreg_off(m->rt2)));
            }
        }
        /* ldp/stp/ldpsw rt, rt2, [x27] */
        emit(j, m->opc << 30 | 0x29000000 | (m->load ? 1 << 22 : 0) |
                rt2 << 10 | H_TMP2 << 5 | rt);
        if (m->load) {
            if (e1) {
                emit(j, i_str_x(rt, H_ENV, xreg_off(m->rt)));
            }
            if (e2) {
                emit(j, i_str_x(rt2, H_ENV, xreg_off(m->rt2)));
            }
        }
    }
}

/* The inline TLB check of x16 (its miss: sl's slow path), then the access */
static void emit_tlb_access(IJ *j, const MemInfo *m, IJSlow *sl, int accsize)
{
    unsigned pb = j->page_bits;
    int cmp_off = m->load ? offsetof(CPUTLBEntry, addr_read)
                          : offsetof(CPUTLBEntry, addr_write);

    sl->cbnz2 = NULL;
    if (m->akind != A_NONE && m->size) {
        /* atomics, exclusives, ordered: naturally aligned (else the slow path) */
        emit(j, 0x92400000 | (m->size - 1) << 10 | H_TMP0 << 5 | H_TMP1);  /* and x17, x16, #(size-1) */
        sl->cbnz2 = j->ptr;
        emit(j, 0);
    }
    if (m->akind == A_LSE || m->akind == A_CAS || m->akind == A_STX) {
        cmp_off = offsetof(CPUTLBEntry, addr_write);
    }

    /*
     * The entry of the first byte's page (x27), compared (rotated: its flag
     * bits on top) with the page of the last byte: an access crossing pages
     * misses, as in TCG. x16 = the last byte from here on.
     */
    emit(j, 0xA9400000 | ((tlb_fast_off(m->midx) / 8) & 0x7f) << 15 |
            H_TMP2 << 10 | H_ENV << 5 | H_TMP1);            /* ldp x17, x27, [env, #fast] */
    emit(j, 0x8A000000 | 1 << 22 | H_TMP0 << 16 | (pb - CPU_TLB_ENTRY_BITS) << 10 |
            H_TMP1 << 5 | H_TMP1);                          /* and x17, x17, x16, lsr #.. */
    emit(j, 0x8B000000 | H_TMP1 << 16 | H_TMP2 << 5 | H_TMP2);  /* add x27, x27, x17 */
    if (accsize > 1) {
        emit_addi(j, H_TMP0, H_TMP0, accsize - 1);
    }
    emit(j, i_ldr_x(H_TMP1, H_TMP2, cmp_off));
    emit(j, i_ldr_x(H_TMP2, H_TMP2, offsetof(CPUTLBEntry, addend)));
    emit(j, 0x93C00000 | H_TMP1 << 16 | pb << 10 | H_TMP1 << 5 | H_TMP1);  /* ror x17, x17, #pb */
    emit(j, 0xCA000000 | 1 << 22 | H_TMP0 << 16 | pb << 10 |
            H_TMP1 << 5 | H_TMP1);                          /* eor x17, x17, x16, lsr #pb */
    sl->cbnz = j->ptr;
    emit(j, 0);                                             /* cbnz x17, slow */
    {
        static int allslow = -1;

        if (allslow < 0) {
            allslow = getenv("ISHJIT_ALLSLOW") != NULL;
        }
        if (allslow) {
            sl->allslow = true;
        }
    }
    emit(j, 0x8B000000 | H_TMP0 << 16 | H_TMP2 << 5 | H_TMP2);  /* add x27, x27, x16 */

    emit_access(j, m, accsize);
}

/*
 * The access at [x27, #off] (x27 the host address of the base), the offset
 * in the instruction; false if it does not fit (then: emit_access)
 */
static bool emit_access_off(IJ *j, const MemInfo *m, int64_t off)
{
    int esz = 1 << m->size;                         /* (one register's bytes) */
    bool scaled = off >= 0 && off % esz == 0 && off / esz < 4096;
    bool unscaled = off >= -256 && off <= 255;
    bool p7 = off % esz == 0 && off / esz >= -64 && off / esz <= 63;

    if (m->akind != A_NONE) {
        return false;
    }
    if (m->vec && !m->pair) {
        unsigned sz = m->size == 4 ? 0 : m->size;
        unsigned opcv = (m->size == 4 ? 2 : 0) | m->load;

        if (scaled) {           /* ldr/str Vt, [x27, #off] */
            emit(j, sz << 30 | 0x3D000000 | opcv << 22 | (off / esz) << 10 |
                    H_TMP2 << 5 | m->rt);
        } else if (unscaled) {  /* ldur/stur */
            emit(j, sz << 30 | 0x3C000000 | opcv << 22 | (off & 0x1ff) << 12 |
                    H_TMP2 << 5 | m->rt);
        } else {
            return false;
        }
        return true;
    }
    if (m->vec) {
        if (!p7) {
            return false;
        }
        /* ldp/stp St/Dt/Qt, [x27, #off] */
        emit(j, (m->size - 2) << 30 | 0x2D000000 | (m->load ? 1 << 22 : 0) |
                ((off / esz) & 0x7f) << 15 | m->rt2 << 10 | H_TMP2 << 5 | m->rt);
        return true;
    }
    if (!m->pair) {
        int rt = m->rt == 31 ? 31 : g2h[m->rt];
        bool in_env = rt == H_NONE;
        uint32_t op;

        if (scaled) {
            op = m->size << 30 | 0x39000000 | m->opc << 22 | (off / esz) << 10;
        } else if (unscaled) {
            op = m->size << 30 | 0x38000000 | m->opc << 22 | (off & 0x1ff) << 12;
        } else {
            return false;
        }
        if (in_env) {
            rt = H_TMP1;
            if (!m->load) {
                emit(j, i_ldr_x(rt, H_ENV, xreg_off(m->rt)));
            }
        }
        emit(j, op | H_TMP2 << 5 | rt);
        if (in_env && m->load) {
            emit(j, i_str_x(rt, H_ENV, xreg_off(m->rt)));
        }
        return true;
    }
    if (!p7) {
        return false;
    }
    {
        int rt = m->rt == 31 ? 31 : g2h[m->rt];
        int rt2 = m->rt2 == 31 ? 31 : g2h[m->rt2];
        bool e1 = rt == H_NONE, e2 = rt2 == H_NONE;

        if (e1) {
            rt = H_TMP0;
        }
        if (e2) {
            rt2 = H_TMP1;
        }
        if (!m->load) {
            if (e1) {
                emit(j, i_ldr_x(rt, H_ENV, xreg_off(m->rt)));
            }
            if (e2) {
                emit(j, i_ldr_x(rt2, H_ENV, xreg_off(m->rt2)));
            }
        }
        /* ldp/stp/ldpsw rt, rt2, [x27, #off] */
        emit(j, m->opc << 30 | 0x29000000 | (m->load ? 1 << 22 : 0) |
                ((off / esz) & 0x7f) << 15 | rt2 << 10 | H_TMP2 << 5 | rt);
        if (m->load) {
            if (e1) {
                emit(j, i_str_x(rt, H_ENV, xreg_off(m->rt)));
            }
            if (e2) {
                emit(j, i_str_x(rt2, H_ENV, xreg_off(m->rt2)));
            }
        }
        return true;
    }
}

/*
 * A grouped access's out-of-line part: the softmmu way (the group's check
 * missed, or its slot is 0); the first access also clears the slot
 */
static void emit_group_cold(IJ *j, IJSlow *sl)
{
    const MemInfo *m = &sl->m;
    int accsize = (1 << m->size) * (m->pair ? 2 : 1);

    (void)accsize;
    if (sl->gkind == 1) {
        emit(j, i_str_x(31, H_ENV, offsetof(CPUARMState, ishjit_gslot[sl->gslot])));  /* (xzr) */
    }
    emit_addr(j, m, sl->ipc);
}

static void emit_mem_(IJ *j, const MemInfo *m, vaddr ipc, IJSlow *sl);
static void emit_mem(IJ *j, const MemInfo *m, vaddr ipc, IJSlow *sl)
{
    intptr_t o0 = ij_offset(j);

    emit_mem_(j, m, ipc, sl);
    ij_region(j, o0, j->g_role == 2 ? 'g' : j->g_role == 1 ? 'G' : 'M');
}

static void emit_mem_(IJ *j, const MemInfo *m, vaddr ipc, IJSlow *sl)
{
    int accsize = (1 << m->size) * (m->pair ? 2 : 1);

    sl->gkind = 0;
    if (j->g_role == 1) {
        /*
         * The group's first access: one TLB check for the whole span of the
         * group's accesses (one page), the base's host address to its slot
         */
        unsigned pb = j->page_bits;
        int cmp_off = j->g_store ? offsetof(CPUTLBEntry, addr_write)
                                 : offsetof(CPUTLBEntry, addr_read);
        int base = g2h[m->rn];
        unsigned soff = offsetof(CPUARMState, ishjit_gslot[j->g_slot]);

        if (base == H_NONE) {
            emit(j, i_ldr_x(H_TMP0, H_ENV, xreg_off(m->rn)));
            base = H_TMP0;
        }
        emit_addi(j, H_TMP0, base, j->g_min);
        emit(j, 0xA9400000 | ((tlb_fast_off(m->midx) / 8) & 0x7f) << 15 |
                H_TMP2 << 10 | H_ENV << 5 | H_TMP1);        /* ldp x17, x27, [env, #fast] */
        emit(j, 0x8A000000 | 1 << 22 | H_TMP0 << 16 | (pb - CPU_TLB_ENTRY_BITS) << 10 |
                H_TMP1 << 5 | H_TMP1);                      /* and x17, x17, x16, lsr #.. */
        emit(j, 0x8B000000 | H_TMP1 << 16 | H_TMP2 << 5 | H_TMP2);  /* add x27, x27, x17 */
        emit_addi(j, H_TMP0, H_TMP0, j->g_span - 1);
        emit(j, i_ldr_x(H_TMP1, H_TMP2, cmp_off));
        emit(j, i_ldr_x(H_TMP2, H_TMP2, offsetof(CPUTLBEntry, addend)));
        emit(j, 0x93C00000 | H_TMP1 << 16 | pb << 10 | H_TMP1 << 5 | H_TMP1);  /* ror x17, x17, #pb */
        emit(j, 0xCA000000 | 1 << 22 | H_TMP0 << 16 | pb << 10 |
                H_TMP1 << 5 | H_TMP1);                      /* eor x17, x17, x16, lsr #pb */
        sl->gbr = j->ptr;
        emit(j, 0xB5000000 | H_TMP1);                       /* cbnz x17, out of line */
        emit(j, 0x8B000000 | H_TMP0 << 16 | H_TMP2 << 5 | H_TMP2);  /* add x27, x27, x16 */
        emit_addi(j, H_TMP2, H_TMP2, -(int64_t)(j->g_span - 1) - j->g_min);
        emit(j, i_str_x(H_TMP2, H_ENV, soff));
        if (!emit_access_off(j, m, m->imm)) {
            emit_addi(j, H_TMP2, H_TMP2, m->imm + accsize - 1);
            emit_access(j, m, accsize);
        }
        sl->gkind = 1;
    } else if (j->g_role == 2) {
        /* a later access of the group: through the slot (0: the softmmu way) */
        emit(j, i_ldr_x(H_TMP2, H_ENV, offsetof(CPUARMState, ishjit_gslot[j->g_slot])));
        sl->gbr = j->ptr;
        emit(j, 0xB4000000 | H_TMP2);                       /* cbz x27, out of line */
        if (!emit_access_off(j, m, m->imm)) {
            emit_addi(j, H_TMP2, H_TMP2, m->imm + accsize - 1);
            emit_access(j, m, accsize);
        }
        sl->gkind = 2;
    } else {
        emit_addr(j, m, ipc);
        emit_tlb_access(j, m, sl, accsize);
    }
    sl->gslot = j->g_slot;
    sl->ipc = ipc;
    sl->resume = j->ptr;

    if (m->wb != WB_NONE) {
        int b = g2h[m->rn];

        if (b == H_NONE) {
            emit(j, i_ldr_x(H_TMP0, H_ENV, xreg_off(m->rn)));
            emit_addi(j, H_TMP0, H_TMP0, m->imm);
            emit(j, i_str_x(H_TMP0, H_ENV, xreg_off(m->rn)));
        } else {
            emit_addi(j, b, b, m->imm);
        }
    }
}

/*
 * The block's slow-path helpers: save stores every guest register the block
 * keeps in host registers (and NZCV, for the flags variant), so that QEMU's
 * load/store helpers (which may raise an exception and leave the block) see
 * the guest state; it also moves the address (x16) to x1. Restore reloads
 * them. Both return to x27.
 */
static void emit_save_thunk(IJ *j, bool flags)
{
    emit_vstore(j, j->vused);
    /* (x16 the address, x27 the descriptor: the routine's address in x17) */
    emit_env_far(j, true, H_TMP1, H_TMP1, routine_off(ISHJIT_R_SLOW));
    emit(j, i_br(H_TMP1));
}

/* (after R_SLOW: the V registers back, then on at x27) */
static void emit_restore_thunk(IJ *j, bool flags)
{
    emit_vload(j, j->vused);
    emit(j, i_br(H_TMP2));
}

static const void *ld_helper(int size, bool sign)
{
    static const void * const u[4] = {
        helper_ldub_mmu, helper_lduw_mmu, helper_ldul_mmu, helper_ldq_mmu,
    };
    static const void * const sx[3] = {
        helper_ldsb_mmu, helper_ldsw_mmu, helper_ldsl_mmu,
    };
    return sign ? sx[size] : u[size];
}

static const void *st_helper(int size)
{
    static const void * const h[4] = {
        helper_stb_mmu, helper_stw_mmu, helper_stl_mmu, helper_stq_mmu,
    };
    return h[size];
}

/*
 * A slow path: x16 the access's first byte, then a branch to R_SLOW (through
 * the TB's save thunk) with x27 = a descriptor of the access (data words
 * after the branch, never executed): ishjit_slow_c does it with QEMU's
 * helpers. Descriptor: [0] what (IJD_*), [1] the resume address (the
 * helpers' return address for unwinding), relative to the descriptor,
 * [2] the TB's V register reload thunk (relative) or 0, [3] the insn
 * (atomics).
 */
#define IJD_SIZE(w)     ((w) & 7)
#define IJD_LOAD        (1u << 3)
#define IJD_PAIR        (1u << 4)
#define IJD_SIGN        (1u << 5)
#define IJD_DEST32      (1u << 6)
#define IJD_VEC         (1u << 7)
#define IJD_AKIND(w)    (((w) >> 8) & 7)
#define IJD_RT(w)       (((w) >> 11) & 31)
#define IJD_RT2(w)      (((w) >> 16) & 31)
#define IJD_MIDX(w)     (((w) >> 21) & 15)
#define IJD_TBID(w)     (((w) >> 25) & 3)
#define IJD_LIT         (1u << 27)

static void emit_slow_desc(IJ *j, const IJSlow *sl, uint32_t *save, uint32_t *restore)
{
    const MemInfo *m = &sl->m;
    uint32_t *adr = j->ptr, *d;
    bool atomic = m->akind == A_LSE || m->akind == A_CAS || m->akind == A_LDX ||
                  m->akind == A_STX;

    g_assert(m->midx >= 0 && m->midx < 16 && m->size <= 4);
    emit(j, 0);                                     /* adr x27, desc */
    emit(j, i_b(save - j->ptr));
    d = j->ptr;
    *adr = i_adr(H_TMP2, (d - adr) * 4);
    emit(j, m->size | (m->load ? IJD_LOAD : 0) | (m->pair ? IJD_PAIR : 0) |
            (m->sign ? IJD_SIGN : 0) | (m->dest32 ? IJD_DEST32 : 0) |
            (m->vec ? IJD_VEC : 0) | m->akind << 8 | m->rt << 11 | m->rt2 << 16 |
            m->midx << 21 | (j->tbid & 3) << 25 | (m->mode == M_LIT ? IJD_LIT : 0));
    ij_fixup(FX_C2H_REL, j->ptr, sl->resume, d);
    emit(j, 0);                                     /* (resume - desc) */
    emit(j, j->vused ? (uint32_t)((restore - d) * 4) : 0);
    if (atomic) {
        emit(j, sl->insn);
    }
}

/* (cold) the stub; *after_sub: where the alignment check's miss enters */
static void emit_slow(IJ *j, const IJSlow *sl, uint32_t *save, uint32_t *restore,
                      uint32_t **after_sub)
{
    const MemInfo *m = &sl->m;
    int accsize = (1 << m->size) * (m->pair ? 2 : 1);

    if (accsize > 1) {
        emit_addi(j, H_TMP0, H_TMP0, -(accsize - 1));
    }
    /* (the alignment check comes here, x16 already the first byte) */
    *after_sub = j->ptr;
    emit_slow_desc(j, sl, save, restore);
}

static uint64_t ij_tbi(uint64_t a, int tbid)
{
    bool hi = (a >> 55) & 1;

    if (tbid == 3 || (tbid == 1 && !hi) || (tbid == 2 && hi)) {
        return sextract64(a, 0, 56);
    }
    return a;
}

IJSlowRet ishjit_slow_c(CPUARMState *env, uint64_t addr, const uint32_t *desc)
{
    uint32_t w = desc[0];
    int size = IJD_SIZE(w), midx = IJD_MIDX(w), akind = IJD_AKIND(w);
    int rt = IJD_RT(w), rt2 = IJD_RT2(w);
    uintptr_t ra = (uintptr_t)desc + (int32_t)desc[1];
    IJSlowRet r = { ra, desc[2] ? (uintptr_t)desc + (int32_t)desc[2] : 0 };

    if (!(w & IJD_LIT)) {
        addr = ij_tbi(addr, IJD_TBID(w));
    }
    if (akind == A_LSE || akind == A_CAS || akind == A_LDX || akind == A_STX) {
        ishjit_atomic_slow(env, addr, desc[3], ra, midx);
        return r;
    }
    if (akind != A_NONE) {
        smp_mb();                                   /* (ordered access) */
    }
    if (w & IJD_VEC) {
        /* through env's image of the V registers, 8 bytes at a time */
        int hsz = MIN(size, 3);
        MemOpIdx voi = make_memop_idx(hsz | MO_LE, midx);

        for (int k = 0; k < ((w & IJD_PAIR) ? 2 : 1); k++) {
            int v = k ? rt2 : rt;

            for (int c = 0; c < (size == 4 ? 2 : 1); c++) {
                uint64_t *p = (uint64_t *)((uint8_t *)env + vreg_off(v) + 8 * c);
                uint64_t a = addr + k * (1 << size) + 8 * c;

                if (w & IJD_LOAD) {
                    uint64_t (*ld)(CPUArchState *, uint64_t, MemOpIdx, uintptr_t) =
                        ld_helper(hsz, false);

                    p[0] = ld(env, a, voi, ra);
                    if (size < 4) {
                        p[1] = 0;
                    }
                } else {
                    void (*st)(CPUArchState *, uint64_t, uint64_t, MemOpIdx, uintptr_t) =
                        st_helper(hsz);

                    st(env, a, p[0], voi, ra);
                }
            }
        }
        return r;
    }
    {
        MemOpIdx oi = make_memop_idx(size | MO_LE | ((w & IJD_SIGN) ? MO_SIGN : 0), midx);
        int n = (w & IJD_PAIR) ? 2 : 1;
        uint64_t v[2];

        for (int k = 0; k < n; k++) {
            uint64_t a = addr + k * (1 << size);
            int reg = k ? rt2 : rt;

            if (w & IJD_LOAD) {
                uint64_t (*ld)(CPUArchState *, uint64_t, MemOpIdx, uintptr_t) =
                    ld_helper(size, w & IJD_SIGN);

                v[k] = ld(env, a, oi, ra);
                if (w & IJD_DEST32) {
                    v[k] = (uint32_t)v[k];
                }
            } else {
                void (*st)(CPUArchState *, uint64_t, uint64_t, MemOpIdx, uintptr_t) =
                    st_helper(size);

                st(env, a, reg == 31 ? 0 : env->xregs[reg], oi, ra);
            }
        }
        if (w & IJD_LOAD) {
            for (int k = 0; k < n; k++) {
                int reg = k ? rt2 : rt;

                if (reg != 31) {
                    env->xregs[reg] = v[k];
                }
            }
        }
    }
    return r;
}


/* debugging: ISHJIT_STATS=1 prints what stops blocks at exit */
static uint64_t st_blocks, st_insns, st_declined;
static uint64_t st_stop[64];
static const char *st_name(int c)
{
    static const char *names[] = {
        [0] = "other", [1] = "fp/simd data", [2] = "fp/simd ld/st",
        [3] = "ld/st exclusive/ordered", [4] = "atomics (lse)",
        [5] = "system/barrier/msr/mrs", [6] = "exception (svc/brk...)",
        [7] = "ldtr/sttr", [8] = "pauth/other branch", [9] = "eret",
        [10] = "end of page/max",
    };
    return c < 11 && names[c] ? names[c] : "?";
}
static int st_class(uint32_t insn)
{
    if (!insn) return 10;
    if ((insn & 0x0E000000) == 0x0E000000 || (insn & 0x1F000000) == 0x1E000000) return 1;
    if ((insn & 0x0A000000) == 0x08000000 && (insn & (1u << 26))) return 2;
    if ((insn & 0x3F000000) == 0x08000000) return 3;
    if ((insn & 0x3B200C00) == 0x38200000) return 4;
    if ((insn & 0x3B200C00) == 0x38000800) return 7;
    if ((insn & 0xFFC00000) == 0xD5000000) return 5;
    if ((insn & 0xFF000000) == 0xD4000000) return 6;
    if (insn == 0xD69F03E0) return 9;
    if ((insn & 0xFE000000) == 0xD6000000) return 8;
    return 0;
}
static void st_dump(void)
{
    fprintf(stderr, "ISHJIT blocks %llu insns %llu (%.1f/block) declined %llu\n",
            (unsigned long long)st_blocks, (unsigned long long)st_insns,
            st_blocks ? (double)st_insns / st_blocks : 0, (unsigned long long)st_declined);
    for (int c = 0; c < 11; c++) {
        if (st_stop[c]) {
            fprintf(stderr, "ISHJIT stop %-28s %llu\n", st_name(c), (unsigned long long)st_stop[c]);
        }
    }
}
static void ishjit_stat(int n, uint32_t stop, bool declined)
{
    static bool init;
    if (!init) { init = true; atexit(st_dump); }
    if (declined) st_declined++; else { st_blocks++; st_insns += n; }
    if (stop || n) st_stop[st_class(stop)] += stop || !n ? 1 : 0;
}


/* sp_el0 is UNDEF while PSTATE.SP is 0 (SP_EL0 is the stack pointer) */
static void emit_spcheck(IJ *j, vaddr ipc)
{
    uint32_t *ok;

    emit(j, i_ldr_w(H_TMP1, H_ENV, offsetof(CPUARMState, pstate)));
    ok = j->ptr;
    emit(j, 0);                                     /* tbnz w17, #0, ok */
    emit_vstore(j, j->vwritten);
    emit_sync(j);
    emit_pc_set(j, ipc);
    emit(j, i_mov_x(0, H_ENV));
    emit_movi(j, 1, EXCP_UDEF);
    emit_movi(j, 2, syn_uncategorized());
    emit_movi(j, H_TMP0, (uintptr_t)helper_exception_with_syndrome);
    emit(j, i_blr(H_TMP0));
    *ok = 0x37000000 | 0 << 19 | ((j->ptr - ok) & 0x3fff) << 5 | H_TMP1;
}



/*
 * An exception taken from generated code without leaving it: what
 * raise_exception() and then cpu_exec's cpu_handle_exception() do (the
 * exception fields, do_interrupt under the BQL), minus the longjmp and the
 * trip through cpu_exec. The caller stored the registers and set env->pc
 * to the return address, then continues at the vector (emit_continue_or_exit).
 */
void ishjit_take_exception(CPUARMState *env, uint32_t excp, uint32_t syn);
void ishjit_take_exception(CPUARMState *env, uint32_t excp, uint32_t syn)
{
    CPUState *cs = env_cpu(env);

    cs->exception_index = excp;
    env->exception.syndrome = syn;
    env->exception.target_el = exception_target_el(env);
    bql_lock();
    cs->cc->tcg_ops->do_interrupt(cs);
    /*
     * do_interrupt asks cpu_exec not to chain from the last TB (the TB
     * flags changed): the caller looks the next TB up afresh instead
     */
    cs->interrupt_request &= ~CPU_INTERRUPT_EXITTB;
    bql_unlock();
    cs->exception_index = -1;
}

/*
 * After a helper that may unmask interrupts (ERET, DAIFClr), registers
 * stored, env->pc the next insn's: back to cpu_exec only with an interrupt
 * pending (as TCG's exit), else straight to the next TB (its full entry).
 */
static void emit_continue_or_exit(IJ *j)
{
    int ir_off = (int)offsetof(ArchCPU, parent_obj.interrupt_request) -
                 (int)offsetof(ArchCPU, env);
    uint32_t *pending;

    emit_movi(j, H_TMP0, (uint64_t)(int64_t)ir_off);
    emit(j, 0xB8606800 | H_TMP0 << 16 | H_ENV << 5 | H_TMP0);   /* ldr w16, [env, x16] */
    pending = j->ptr;
    emit(j, 0);                                     /* cbnz w16, exit */
    emit(j, i_mov_x(0, H_ENV));
    emit_movi(j, H_TMP0, (uintptr_t)helper_lookup_tb_ptr);
    emit(j, i_blr(H_TMP0));
    emit(j, i_br(0));
    *pending = 0x35000000 | ((j->ptr - pending) & 0x7ffff) << 5 | H_TMP0;
    emit_movi(j, H_TMP0, (uintptr_t)tcg_code_gen_epilogue);
    emit(j, i_br(H_TMP0));
}

static int ishjit_gen(CPUState *cs, TranslationBlock *tb, vaddr pc,
                      void *host_pc, int max_insns, void *buf)
{
    CPUARMState *env = cpu_env(cs);
    IJ jj = { .ptr = buf, .start = buf, .tb = tb, .env = env,
              .pc_first = pc, .highwater = tcg_ctx->code_gen_highwater };
    IJ *j = &jj;
    uint32_t insns[TCG_MAX_INSNS];
    OperandInfo ops[TCG_MAX_INSNS];
    MemInfo mems[TCG_MAX_INSNS];
    SimdOperands simds[TCG_MAX_INSNS];
    IJSlow slows[TCG_MAX_INSNS];
    bool fwritten[TCG_MAX_INSNS];       /* NZCV written by an earlier insn */
    uint32_t written[TCG_MAX_INSNS];    /* (each insn's written registers) */
    int16_t glead[TCG_MAX_INSNS];        /* (shared checks: the group's first access, or -1) */
    int8_t gslot[TCG_MAX_INSNS];
    int64_t gmin[TCG_MAX_INSNS], gend[TCG_MAX_INSNS];
    int gspan[TCG_MAX_INSNS];
    bool gstore[TCG_MAX_INSNS];
    int kinds[TCG_MAX_INSNS];
    int n = 0, last;
    intptr_t slow0 = 0;
    uint32_t *full_b = NULL;
    CPUARMTBFlags tbf = arm_tbflags_from_tb(tb);
    vaddr page_end = (pc & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
    /*
     * The per-insn unwind data for encode_search: TCG allocates its own
     * from the translation's pool (sized for its TB), so ishjit has its own
     * per thread, for the largest TB.
     */
    static __thread uint64_t *ij_insn_data;
    uint64_t *data;
    uint16_t *end_off = tcg_ctx->gen_insn_end_off;

    if (!ij_insn_data) {
        ij_insn_data = g_new0(uint64_t, TCG_MAX_INSNS * TARGET_INSN_START_WORDS);
    }
    ij_nreg = 0;
    ij_nfix = 0;
    j->cptr = ij_cold_buf;
    data = ij_insn_data;
    tcg_ctx->gen_insn_data = data;

    /* Memory accesses: through the TB's MMU index, unless checks apply */
    j->mmu_idx = EX_TBFLAG_ANY(tbf, MMUIDX);
    j->tbid = EX_TBFLAG_A64(tbf, TBID);
    j->page_bits = TARGET_PAGE_BITS;
    g_assert(j->page_bits == shared_page_bits && TARGET_PAGE_BITS <= 16);
    if (lookup_shared < 0) {
        lookup_shared = !getenv("ISHJIT_INLINE_LOOKUP");
    }
    j->el = arm_current_el(env);
    j->fp_ok = EX_TBFLAG_ANY(tbf, FPEXC_EL) == 0;
    j->fp_default = EX_TBFLAG_A64(tbf, FPDEFAULT);
    j->fgt_active = EX_TBFLAG_ANY(tbf, FGT_ACTIVE);
    j->fgt_svc = EX_TBFLAG_ANY(tbf, FGT_SVC);
    j->trap_eret = EX_TBFLAG_A64(tbf, TRAP_ERET);
    j->unpriv = EX_TBFLAG_A64(tbf, UNPRIV);
    j->el2_el3 = arm_feature(env, ARM_FEATURE_EL2) || arm_feature(env, ARM_FEATURE_EL3);
    static int nomem = -1;
    if (nomem < 0) {
        nomem = getenv("ISHJIT_NOMEM") != NULL;
    }
    j->mem_ok = !nomem && !EX_TBFLAG_ANY(tbf, ALIGN_MEM) &&
                !EX_TBFLAG_A64(tbf, MTE_ACTIVE) &&
                !EX_TBFLAG_A64(tbf, MTE0_ACTIVE) &&
                tlb_fast_off(j->mmu_idx) >= -512 &&
                tlb_fast_off(j->mmu_idx) % 8 == 0;

    /* Scan: the run of instructions ishjit translates, up to a branch */
    while (n < max_insns && pc + 4 * n < page_end) {
        uint32_t insn = ldl_le_p((uint8_t *)host_pc + 4 * n);
        uint32_t rd = 0, wr = 0;
        int k = classify(insn, &ops[n], &mems[n], j->mem_ok, j->fp_ok,
                         &simds[n], j);

        if (k == K_STOP) {
            break;
        }
        insns[n] = insn;
        kinds[n] = k;
        fwritten[n] = j->flags_written;
        switch (k) {
        case K_FP: {
            uint32_t v;

            fp_regs(insn, &simds[n], &rd, &wr, &v);
            j->vused |= v;
            j->vwritten |= v;
            break;
        }
        case K_MRS:
        case K_MRSCONST:
            wr = (insn & 31) == 31 ? 0 : 1u << (insn & 31);
            break;
        case K_MSR:
            rd = (insn & 31) == 31 ? 0 : 1u << (insn & 31);
            break;
        case K_MEM:
            if (mems[n].akind == A_STX && mems[n].rt != 31 &&
                g2h[mems[n].rt] == H_NONE) {
                k = K_STOP;                 /* (no scratch left for it) */
                break;
            }
            mem_regs(&mems[n], &rd, &wr);
            if (mems[n].vec) {
                uint32_t v = 1u << mems[n].rt | (mems[n].pair ? 1u << mems[n].rt2 : 0);

                j->vused |= v;
                if (mems[n].load) {
                    j->vwritten |= v;
                }
            }
            break;
        case K_DP:
            operand_regs(insn, &ops[n], &rd, &wr);
            if (ops[n].reads_flags && !j->flags_written) {
                j->flags_in = true;
            }
            if (ops[n].writes_flags) {
                j->flags_written = j->flags_out = true;
            }
            break;
        case K_ADR:
            wr = (insn & 31) == 31 ? 0 : 1u << (insn & 31);
            break;
        case K_BL:
            wr = 1u << 30;
            break;
        case K_BCOND:
            if (!j->flags_written) {
                j->flags_in = true;
            }
            break;
        case K_CB:
        case K_TB:
            rd = (insn & 31) == 31 ? 0 : 1u << (insn & 31);
            break;
        case K_BREG:
            rd = ((insn >> 5) & 31) == 31 ? 0 : 1u << ((insn >> 5) & 31);
            if (((insn >> 21) & 3) == 1) {
                wr = 1u << 30;                  /* blr */
            }
            break;
        }
        if (k == K_STOP) {
            break;
        }
        /* env-held registers: at most three per DP insn, none for cb/tb */
        {
            uint32_t mem = (rd | wr) & ENV_HELD;

            if (((k == K_DP || k == K_FP) && ctpop32(mem) > 3) ||
                ((k == K_CB || k == K_TB) && mem)) {
                break;
            }
        }
        j->used |= rd | wr;
        j->written |= wr;
        written[n] = wr;
        n++;
        if (k >= K_B) {
            break;                              /* a branch ends the block */
        }
    }
    static int stats = -1;
    if (stats < 0) {
        stats = getenv("ISHJIT_STATS") != NULL;
    }
    if (stats) {
        ishjit_stat(n, n < max_insns && pc + 4 * n < page_end && (n == 0 || kinds[n - 1] < K_B) ?
                    ldl_le_p((uint8_t *)host_pc + 4 * n) : 0, n == 0);
    }
    if (n == 0) {
        return ISHJIT_DECLINE;
    }
    /*
     * Stopped short by an instruction ishjit does not translate: the rest
     * of the basic block would be a TCG TB, and the switch between the
     * two (all guest registers stored, then loaded) costs more than a few
     * translated instructions save, every time through a loop. Unless the
     * part translated is long, the whole block is TCG's.
     */
    if (n < 16 && n < max_insns && pc + 4 * n < page_end &&
        kinds[n - 1] < K_B) {
        return ISHJIT_DECLINE;
    }
    last = n - 1;

    /* The TCG TB prologue's check: exit if an interrupt or exit is pending */
    {
        int off = (int)offsetof(ArchCPU, parent_obj.neg.icount_decr.u32) -
                  (int)offsetof(ArchCPU, env);
        uint32_t *tbnz;

        /*
         * The full entry (tc.ptr) jumps to its part at the TB's end (the
         * registers loaded), then back to the fast entry. Before the fast
         * entry: a near jump to the far exit (a TB can be beyond tbnz's
         * 32KB reach), so that the fast entry's check is a single tbnz.
         */
        full_b = j->ptr;
        emit(j, 0);                             /* b (full entry) */
        tbnz = j->ptr;
        emit(j, 0);                             /* b exit */
        tb->ishjit_fast = ij_offset(j);
        emit(j, i_ldur_w(H_TMP0, H_ENV, off));
        emit(j, 0x37000000 | 31 << 19 | ((tbnz - j->ptr) & 0x3fff) << 5 | H_TMP0);
        emit_checkenv(j, 1);
        emit_vload(j, j->vused);
        ij_region(j, 0, 'E');
        /* (the exit it jumps to is placed after the block's code) */
        tb->jmp_insn_offset[0] = TB_JMP_OFFSET_INVALID;
        tb->jmp_insn_offset[1] = TB_JMP_OFFSET_INVALID;
        tb->jmp_reset_offset[0] = TB_JMP_OFFSET_INVALID;
        tb->jmp_reset_offset[1] = TB_JMP_OFFSET_INVALID;

        /*
         * Shared TLB checks: accesses through the same base register, its
         * value unchanged in between (only data processing, FP/SIMD and
         * other accesses between them), immediate offsets, no writeback:
         * up to 8 groups, each within 4 KB
         */
        {
            static int nogroup = -1;
            int ngroups = 0;

            if (nogroup < 0) {
                nogroup = getenv("ISHJIT_NOGROUP") != NULL;
            }
            for (int i = 0; i < n; i++) {
                const MemInfo *m = &mems[i];
                int acc = (1 << m->size) * (m->pair ? 2 : 1);

                glead[i] = -1;
                if (nogroup || kinds[i] != K_MEM || m->akind != A_NONE ||
                    m->mode != M_IMM || m->wb != WB_NONE) {
                    continue;
                }
                for (int q = i - 1; q >= 0; q--) {
                    const MemInfo *mq = &mems[q];

                    if ((written[q] >> m->rn & 1) ||
                        (kinds[q] != K_DP && kinds[q] != K_MEM && kinds[q] != K_FP &&
                         kinds[q] != K_NOP && kinds[q] != K_ADR)) {
                        break;
                    }
                    if (kinds[q] == K_MEM && mq->akind == A_NONE && mq->mode == M_IMM &&
                        mq->wb == WB_NONE && mq->rn == m->rn && mq->midx == m->midx) {
                        int L = glead[q];
                        int64_t lo, hi;

                        if (L < 0) {
                            if (ngroups == ARRAY_SIZE(((CPUARMState *)0)->ishjit_gslot)) {
                                break;
                            }
                            L = q;
                            gmin[L] = mq->imm;
                            gend[L] = mq->imm + (1 << mq->size) * (mq->pair ? 2 : 1);
                            gstore[L] = !mq->load;
                        }
                        lo = MIN(gmin[L], m->imm);
                        hi = MAX(gend[L], m->imm + acc);
                        if (hi - lo > 4096) {
                            break;
                        }
                        if (glead[q] < 0) {
                            glead[q] = q;
                            gslot[q] = ngroups++;
                        }
                        gmin[L] = lo;
                        gend[L] = hi;
                        gstore[L] |= !m->load;
                        glead[i] = L;
                        break;
                    }
                }
            }
            for (int i = 0; i < n; i++) {
                if (glead[i] == i) {
                    gspan[i] = gend[i] - gmin[i];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            uint32_t insn = insns[i];
            vaddr ipc = j->pc_first + 4 * i;

            if ((uint8_t *)j->ptr + 1024 > j->highwater) {
                return -1;                      /* code buffer full */
            }
            j->cur_insn = insn;
            data[i * TARGET_INSN_START_WORDS + 0] = ipc & ~TARGET_PAGE_MASK;
            emit_checkenv(j, 0x100 + i);
            for (int w = 1; w < TARGET_INSN_START_WORDS; w++) {
                data[i * TARGET_INSN_START_WORDS + w] = 0;
            }

            switch (kinds[i]) {
            case K_NOP:
                break;
            case K_COPY:
                /*
                 * dmb: the host's own; dsb: a dmb ish too, as TCG (what a
                 * dsb waits for, TLB and device operations, is emulated: the
                 * host's dsb would only cost hundreds of cycles)
                 */
                if ((insn & 0xFFFFF0FF) == 0xD503309F) {
                    emit(j, 0xD5033BBF);            /* dmb ish */
                } else {
                    emit(j, insn);
                }
                break;
            case K_CLREX:
                emit(j, 0x92800000 | H_TMP1);       /* mov x17, #-1 */
                emit(j, i_str_x(H_TMP1, H_ENV, offsetof(CPUARMState, exclusive_addr)));
                break;
            case K_MRS: {
                int g = insn & 31;

                if (ops[i].spcheck) {
                    emit_spcheck(j, ipc);
                }
                if (g == 31) {
                    break;
                }
                if (g2h[g] == H_NONE) {
                    emit(j, i_ldr_x(H_TMP0, H_ENV, ops[i].sysoff));
                    emit(j, i_str_x(H_TMP0, H_ENV, xreg_off(g)));
                } else {
                    emit(j, i_ldr_x(g2h[g], H_ENV, ops[i].sysoff));
                }
                break;
            }
            case K_MRSCONST: {
                int g = insn & 31;

                if (g != 31) {
                    int h = g2h[g] == H_NONE ? H_TMP0 : g2h[g];

                    emit_movi(j, h, ops[i].sysconst);
                    if (g2h[g] == H_NONE) {
                        emit(j, i_str_x(h, H_ENV, xreg_off(g)));
                    }
                }
                break;
            }
            case K_MSR: {
                int g = insn & 31;
                int h = g == 31 ? 31 : g2h[g];

                if (ops[i].spcheck) {
                    emit_spcheck(j, ipc);
                }
                if (h == H_NONE) {
                    h = H_TMP0;
                    emit(j, i_ldr_x(h, H_ENV, xreg_off(g)));
                }
                emit(j, i_str_x(h, H_ENV, ops[i].sysoff));
                break;
            }
            case K_SVC:
            case K_BRK: {
                uint32_t imm = (insn >> 5) & 0xffff;

                end_off[i] = ij_offset(j);
                emit_vstore(j, j->vwritten);
                emit_sync(j);
                /* svc: the exception returns after it; brk: to it */
                emit_pc_set(j, ipc + (kinds[i] == K_SVC ? 4 : 0));
                emit(j, i_mov_x(0, H_ENV));
                if (kinds[i] == K_SVC) {
                    /* taken here, then on at the vector (no cpu_exec trip) */
                    emit_movi(j, 1, EXCP_SWI);
                    emit_movi(j, 2, syn_aa64_svc(imm));
                    emit_movi(j, H_TMP0, (uintptr_t)ishjit_take_exception);
                    emit(j, i_blr(H_TMP0));
                    emit_continue_or_exit(j);
                } else {
                    emit_movi(j, 1, syn_aa64_bkpt(imm));
                    emit_movi(j, H_TMP0, (uintptr_t)helper_exception_bkpt_insn);
                    emit(j, i_blr(H_TMP0));
                    emit(j, 0xD4200000);            /* (noreturn) brk #0 */
                }
                break;
            }
            case K_ERET:
                end_off[i] = ij_offset(j);
                emit_vstore(j, j->vwritten);
                emit_sync(j);
                emit(j, i_ldr_x(1, H_ENV, offsetof(CPUARMState, elr_el[1])));
                emit(j, i_mov_x(0, H_ENV));
                emit_movi(j, H_TMP0, (uintptr_t)helper_exception_return);
                emit(j, i_blr(H_TMP0));
                emit_continue_or_exit(j);
                break;
            case K_ISB:
                end_off[i] = ij_offset(j);
                emit_jump(j, 0, ipc + 4);           /* (ends the TB, as TCG) */
                break;
            case K_DAIFSET:
            case K_DAIFCLR: {
                /*
                 * Inline (no TB flags depend on DAIF; EL1, no traps):
                 * env->daif |= or &= ~ the bits. Clearing them: an
                 * interrupt already pending is taken now, from cpu_exec.
                 */
                uint64_t bits = (uint64_t)((insn >> 8) & 15) << 6;
                uint32_t *none;

                emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, daif)));
                emit_movi(j, H_TMP1, bits);
                emit(j, (kinds[i] == K_DAIFSET ? 0xAA000000 : 0x8A200000) |
                        H_TMP1 << 16 | H_TMP0 << 5 | H_TMP0);   /* orr / bic x16, x16, x17 */
                emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, daif)));
                if (kinds[i] == K_DAIFCLR) {
                    int ir_off = (int)offsetof(ArchCPU, parent_obj.interrupt_request) -
                                 (int)offsetof(ArchCPU, env);

                    emit_movi(j, H_TMP0, (uint64_t)(int64_t)ir_off);
                    emit(j, 0xB8606800 | H_TMP0 << 16 | H_ENV << 5 | H_TMP0);  /* ldr w16, [env, x16] */
                    none = j->ptr;
                    emit(j, 0);                             /* cbz w16, on */
                    emit_vstore(j, j->vwritten);
                    emit_sync(j);
                    emit_pc_set(j, ipc + 4);
                    emit_movi(j, H_TMP0, (uintptr_t)tcg_code_gen_epilogue);
                    emit(j, i_br(H_TMP0));
                    *none = 0x34000000 | ((j->ptr - none) & 0x7ffff) << 5 | H_TMP0;
                }
                break;
            }
            case K_FP: {
                const SimdOperands *g = &simds[i];
                uint32_t rd, wr, vv, mem, out = insn;
                int tmp[32], next = H_TMP0;
                int sh[3] = { 0, 5, 16 };
                uint8_t role[3] = { g->rd, g->rn, g->rm };

                fp_regs(insn, g, &rd, &wr, &vv);
                mem = (rd | wr) & ENV_HELD;
                for (int r = 0; r < 31; r++) {
                    tmp[r] = H_NONE;
                    if ((mem >> r) & 1) {
                        tmp[r] = next;
                        next = next == H_TMP0 ? H_TMP1 : H_TMP2;
                        if ((rd >> r) & 1) {
                            emit(j, i_ldr_x(tmp[r], H_ENV, xreg_off(r)));
                        }
                    }
                }
                for (int k = 0; k < 3; k++) {
                    int r = (insn >> sh[k]) & 31;

                    if (role[k] != SIMD_NOT_GPR && r != 31) {
                        int h = g2h[r] == H_NONE ? tmp[r] : g2h[r];

                        out = (out & ~(0x1Fu << sh[k])) | (uint32_t)h << sh[k];
                    }
                }
                emit(j, out);
                for (int r = 0; r < 31; r++) {
                    if ((wr >> r) & 1 && g2h[r] == H_NONE) {
                        emit(j, i_str_x(tmp[r], H_ENV, xreg_off(r)));
                    }
                }
                break;
            }
            case K_MEM:
                slows[j->nslow].m = mems[i];
                slows[j->nslow].allslow = false;
                slows[j->nslow].insn = insn;
                slows[j->nslow].flags = true;   /* (NZCV always live) */
                j->g_role = glead[i] < 0 ? 0 : glead[i] == i ? 1 : 2;
                j->g_slot = glead[i] < 0 ? 0 : gslot[glead[i]];
                j->g_min = glead[i] < 0 ? 0 : gmin[glead[i]];
                j->g_span = glead[i] < 0 ? 0 : gspan[glead[i]];
                j->g_store = glead[i] >= 0 && gstore[glead[i]];
                emit_mem(j, &mems[i], ipc, &slows[j->nslow]);
                j->nslow++;
                break;
            case K_DP: {
                const OperandInfo *o = &ops[i];
                int tmp[32];
                uint32_t rd, wr, mem;
                int next = H_TMP0;

                operand_regs(insn, o, &rd, &wr);
                mem = (rd | wr) & ENV_HELD;
                for (int g = 0; g < 31; g++) {
                    tmp[g] = H_NONE;
                    if ((mem >> g) & 1) {
                        tmp[g] = next;
                        next = next == H_TMP0 ? H_TMP1 : H_TMP2;
                        if ((rd >> g) & 1) {
                            emit(j, i_ldr_x(tmp[g], H_ENV, xreg_off(g)));
                        }
                    }
                }
                insn &= ~(0x1Fu | 0x1Fu << 5 | 0x1Fu << 16 | 0x1Fu << 10)
                        | (o->rd == F_NONE ? 0x1F : 0)
                        | (o->rn == F_NONE ? 0x1Fu << 5 : 0)
                        | (o->rm == F_NONE ? 0x1Fu << 16 : 0)
                        | (o->ra == F_NONE ? 0x1Fu << 10 : 0);
                if (o->rd != F_NONE) {
                    insn |= host_reg_for(insns[i] & 31, o->rd, tmp);
                }
                if (o->rn != F_NONE) {
                    insn |= host_reg_for((insns[i] >> 5) & 31, o->rn, tmp) << 5;
                }
                if (o->rm != F_NONE) {
                    insn |= host_reg_for((insns[i] >> 16) & 31, o->rm, tmp) << 16;
                }
                if (o->ra != F_NONE) {
                    insn |= host_reg_for((insns[i] >> 10) & 31, o->ra, tmp) << 10;
                }
                emit(j, insn);
                for (int g = 0; g < 31; g++) {
                    if ((wr >> g) & 1 && g2h[g] == H_NONE) {
                        emit(j, i_str_x(tmp[g], H_ENV, xreg_off(g)));
                    }
                }
                break;
            }
            case K_ADR: {
                int g = insn & 31;
                int64_t imm = sextract64(insn, 5, 19) << 2 | ((insn >> 29) & 3);
                int h = g2h[g] == H_NONE ? H_TMP0 : g2h[g];
                vaddr val;

                if (g == 31) {
                    break;                      /* (xzr) */
                }
                if (insn & (1u << 31)) {        /* adrp */
                    val = (ipc & ~(vaddr)0xfff) + (imm << 12);   /* (4 KB, whatever the page size) */
                } else {
                    val = ipc + imm;
                }
                emit_pc_rel(j, h, val);
                if (g2h[g] == H_NONE) {
                    emit(j, i_str_x(h, H_ENV, xreg_off(g)));
                }
                break;
            }
            case K_BL:
            case K_B: {
                vaddr dest = ipc + (sextract64(insn, 0, 26) << 2);

                if (kinds[i] == K_BL) {
                    emit_pc_rel(j, g2h[30], ipc + 4);
                }
                end_off[i] = ij_offset(j);
                emit_jump(j, 0, dest);
                break;
            }
            case K_BCOND:
            case K_CB:
            case K_TB: {
                vaddr dest;
                uint32_t *br;

                if (kinds[i] == K_BCOND || kinds[i] == K_CB) {
                    dest = ipc + (sextract64(insn, 5, 19) << 2);
                } else {
                    dest = ipc + (sextract64(insn, 5, 14) << 2);
                }
                end_off[i] = ij_offset(j);
                /* the taken way: the same branch to a stub further on */
                br = j->ptr;
                emit(j, 0);
                emit_jump(j, 0, ipc + 4);
                {
                    ptrdiff_t d = j->ptr - br;      /* in insns */

                    if (kinds[i] == K_BCOND) {
                        *br = (insn & 0xFF00001F) | (d & 0x7ffff) << 5;
                    } else if (kinds[i] == K_CB) {
                        *br = (insn & 0xFF00001F & ~0x1Fu) |
                              branch_src(j, insn & 31) | (d & 0x7ffff) << 5;
                    } else {
                        *br = (insn & 0xFFF8001F & ~0x1Fu) |
                              branch_src(j, insn & 31) | (d & 0x3fff) << 5;
                    }
                }
                emit_jump(j, 1, dest);
                break;
            }
            case K_BREG: {
                int g = (insn >> 5) & 31;
                int src = branch_src(j, g);

                emit(j, i_mov_x(H_TMP0, src));
                if (((insn >> 21) & 3) == 1) {          /* blr */
                    emit_pc_rel(j, g2h[30], ipc + 4);
                }
                end_off[i] = ij_offset(j);
                emit_lookup(j);                 /* (target in x16) */
                break;
            }
            }
            if (kinds[i] < K_B) {
                end_off[i] = ij_offset(j);
            }
        }

        /* A block not ended by a branch continues at the next insn */
        if (kinds[last] < K_B) {
            emit_jump(j, 0, j->pc_first + 4 * n);
        }

        /*
         * Slow paths: the block's save/restore helpers, then one stub each
         * (cold), and a jump there for each at the hot code's end
         */
        slow0 = ij_offset(j);
        if (j->nslow) {
            uint32_t *save[2] = { NULL, NULL }, *restore[2] = { NULL, NULL };

            for (int f = 0; f < 2; f++) {
                bool need = false;

                for (int k = 0; k < j->nslow; k++) {
                    need |= slows[k].flags == f;
                }
                if (need) {
                    ij_to_cold(j);
                    save[f] = j->ptr;
                    emit_save_thunk(j, f);
                    restore[f] = j->ptr;
                    emit_restore_thunk(j, f);
                    ij_to_hot(j);
                }
            }
            for (int k = 0; k < j->nslow; k++) {
                IJSlow *sl = &slows[k];
                uint32_t *c, *after_sub = NULL, *t;

                if ((uint8_t *)j->ptr + 1024 > j->highwater) {
                    return -1;
                }
                if (j->cptr - ij_cold_buf > IJ_COLD_MAX - 256) {
                    return -2;
                }
                ij_to_cold(j);
                c = j->ptr;
                if (sl->gkind) {
                    emit_group_cold(j, sl);
                    emit_slow_desc(j, sl, save[sl->flags], restore[sl->flags]);
                } else {
                    emit_slow(j, sl, save[sl->flags], restore[sl->flags], &after_sub);
                }
                ij_to_hot(j);
                t = j->ptr;
                emit_b_cold(j, c);
                if (sl->gkind) {
                    *sl->gbr |= ((t - sl->gbr) & 0x7ffff) << 5;
                } else {
                    *sl->cbnz = sl->allslow ? i_b(t - sl->cbnz) :
                                0xB5000000 | ((t - sl->cbnz) & 0x7ffff) << 5 | H_TMP1;
                    if (sl->cbnz2) {
                        t = j->ptr;
                        emit_b_cold(j, after_sub);
                        *sl->cbnz2 = 0xB5000000 | ((t - sl->cbnz2) & 0x7ffff) << 5 | H_TMP1;
                    }
                }
            }
        }

        /* The pending-exit check's target (cold): nothing is live yet */
        ij_to_cold(j);
        {
            uint32_t *c = j->ptr;

            emit_pcsync(j, j->pc_first);
            emit_exit_tb(j, TB_EXIT_REQUESTED);
            ij_to_hot(j);
            *tbnz = 0;
            j->hot_save = j->ptr;
            j->ptr = tbnz;
            emit_b_cold(j, c);
            j->ptr = j->hot_save;
            ij_to_cold(j);
        }
        /* The full entry's part, then the jumps to R_PCSYNC and the exit */
        {
            uint32_t *c = j->ptr;

            ij_to_hot(j);
            j->hot_save = j->ptr;
            j->ptr = full_b;
            emit_b_cold(j, c);
            j->ptr = j->hot_save;
            ij_to_cold(j);
        }
        emit_block_entry(j);
        ij_fixup(FX_C2H, j->ptr, (uint32_t *)((uint8_t *)j->start + tb->ishjit_fast), NULL);
        emit(j, 0);                             /* b (fast entry) */
        if (j->npcsync) {
            for (int k = 0; k < j->npcsync; k++) {
                *j->pend_pcsync[k] = i_b(j->ptr - j->pend_pcsync[k]);
            }
            emit_jump_routine(j, ISHJIT_R_PCSYNC, H_TMP0);
        }
        if (j->nexit) {
            for (int k = 0; k < j->nexit; k++) {
                *j->pend_exit[k] = i_b(j->ptr - j->pend_exit[k]);
            }
            emit_jump_routine(j, ISHJIT_R_EXITTB, H_TMP0);
        }
        ij_to_hot(j);
    }

    if (ij_offset(j) > 60000) {
        tb->ishjit_fast = 0;
        return -2;                              /* (retried with fewer insns) */
    }
    if (ishjit_pmap_get()) {
        FILE *f = ishjit_pmap_get();

        ij_region(j, slow0, 'S');
        flockfile(f);
        fprintf(f, "R %" PRIxPTR, (uintptr_t)tcg_splitwx_to_rx(j->start));
        for (int k = 0; k < ij_nreg; k++) {
            fprintf(f, " %c%x:%x", ij_reg[k].k, ij_reg[k].o0, ij_reg[k].o1);
        }
        fputc('\n', f);
        funlockfile(f);
    }
    /*
     * The cold code below the TB's struct (the top of what the region has
     * left), the branches between hot and cold code fixed up
     */
    {
        size_t cbytes = (j->cptr - ij_cold_buf) * 4;
        uint8_t *cold = (uint8_t *)ROUND_DOWN((uintptr_t)tcg_ctx->code_gen_tbtop - cbytes, 16);
#define COLD(p) ((uint32_t *)(cold + ((uint8_t *)(p) - (uint8_t *)ij_cold_buf)))

        g_assert(!j->in_cold);
        if (j->cptr - ij_cold_buf > IJ_COLD_MAX) {
            tb->ishjit_fast = 0;
            return -2;
        }
        if ((uint8_t *)j->ptr + 1024 + n * 16 + IJ_HIGHWATER > cold) {
            return -1;                          /* (the region is full) */
        }
        memcpy(cold, ij_cold_buf, cbytes);
        for (int k = 0; k < ij_nfix; k++) {
            IJFix *f = &ij_fix[k];

            switch (f->kind) {
            case FX_H2C:
                *f->at = i_b(COLD(f->to) - f->at);
                break;
            case FX_C2H:
                *COLD(f->at) = i_b(f->to - COLD(f->at));
                break;
            case FX_C2H_REL:
                *COLD(f->at) = (uint32_t)((uint8_t *)f->to - (uint8_t *)COLD(f->base));
                break;
            }
        }
        for (int w = 0; w < 2; w++) {
            if (j->jmp2_c[w]) {
                tb->ishjit_jmp2[w] = (uint8_t *)COLD(j->jmp2_c[w]) - (uint8_t *)j->start;
                tb->ishjit_jmp2_reset[w] = (uint8_t *)COLD(j->jmp2r_c[w]) - (uint8_t *)j->start;
            }
        }
#undef COLD
        tcg_ctx->code_gen_tbtop = cold;
        tcg_ctx->code_gen_highwater = cold - IJ_HIGHWATER;
        flush_idcache_range((uintptr_t)tcg_splitwx_to_rx(cold), (uintptr_t)cold, cbytes);
    }
    tb->size = 4 * n;
    tb->icount = n;
    tcg_ctx->data_gen_ptr = NULL;
    /* as tcg_gen_code: the new code reaches the instruction cache */
    flush_idcache_range((uintptr_t)tcg_splitwx_to_rx(j->start),
                        (uintptr_t)j->start, ij_offset(j));
    return ij_offset(j);
}

static int ishjit_gen_code(CPUState *cs, TranslationBlock *tb, vaddr pc,
                    void *host_pc, int max_insns, void *buf)
{
    CPUARMState *env = cpu_env(cs);
    uint32_t cflags = tb_cflags(tb);

    tb->ishjit_fast = 0;                        /* (also when retried) */
    /*
     * The shared routines exist once TCG's prologue is generated, after
     * the CPUs are created: their addresses go to every vCPU (a TB one
     * translates runs on all) before the first TB.
     */
    if (unlikely(!env->ishjit_routine[ISHJIT_R_LOAD])) {
        CPUState *c;

        g_assert(ishjit_routine_rx[ISHJIT_R_LOAD]);
        CPU_FOREACH(c) {
            CPUARMState *e = cpu_env(c);

            ishjit_routine_rx[ISHJIT_R_EXITTB] = (uintptr_t)tcg_ishjit_tb_ret_addr();
            for (int r = 0; r < ISHJIT_NROUTINES; r++) {
                if (r == ISHJIT_R_SCRATCH) {
                    continue;
                }
                qatomic_set(&e->ishjit_routine[r], ishjit_routine_rx[r]);
            }
        }
    }
    tb->ishjit_jmp2[0] = tb->ishjit_jmp2[1] = 0;
    if (!host_pc || !is_a64(env) || (pc & 3) ||
        !(cflags & CF_PCREL) ||
        (cflags & (CF_USE_ICOUNT | CF_NO_GOTO_TB | CF_NO_GOTO_PTR |
                   CF_SINGLE_STEP | CF_MEMI_ONLY)) ||
        EX_TBFLAG_ANY(arm_tbflags_from_tb(tb), SS_ACTIVE)) {
        return ISHJIT_DECLINE;
    }
    /* debugging: ISHJIT_EL0ONLY=1 (user code only), ISHJIT_NOMEM=1 */
    {
        static int el0only = -1;

        if (el0only < 0) {
            el0only = getenv("ISHJIT_EL0ONLY") != NULL;
        }
        if (el0only && arm_current_el(env) != 0) {
            return ISHJIT_DECLINE;
        }
    }
    return ishjit_gen(cs, tb, pc, host_pc, max_insns, buf);
}


/* debugging: ISHJIT_CRASHDUMP=1 dumps registers and code on a host crash */
#include <signal.h>
#include <sys/ucontext.h>
static void cd_hex(uint64_t v)
{
    char b[19] = "0x";
    for (int i = 0; i < 16; i++) {
        b[2 + i] = "0123456789abcdef"[(v >> (60 - 4 * i)) & 15];
    }
    b[18] = ' ';
    if (write(2, b, 19) < 0) { }
}
static void cd_str(const char *s)
{
    if (write(2, s, strlen(s)) < 0) { }
}
static void cd_code(uint64_t pc)
{
    cd_str("\ncode at "); cd_hex(pc); cd_str(":\n");
    for (int i = -24; i < 8; i++) {
        cd_hex(*(uint32_t *)(pc + 4 * i));
        if ((i + 24) % 8 == 7) cd_str("\n");
    }
}
static void cd_handler(int sig, siginfo_t *si, void *uc_)
{
    ucontext_t *uc = uc_;
    cd_str("\nISHJIT CRASH sig "); cd_hex(sig); cd_str("addr "); cd_hex((uint64_t)si->si_addr);
    cd_str("\npc "); cd_hex(uc->uc_mcontext->__ss.__pc);
    cd_str("lr "); cd_hex(uc->uc_mcontext->__ss.__lr);
    cd_str("fp "); cd_hex(uc->uc_mcontext->__ss.__fp);
    cd_str("sp "); cd_hex(uc->uc_mcontext->__ss.__sp);
    for (int i = 0; i < 29; i++) {
        if (i % 4 == 0) cd_str("\n");
        cd_str("x"); { char d[3] = { '0' + i / 10, '0' + i % 10, ' ' }; if (write(2, d, 3) < 0) {} }
        cd_hex(uc->uc_mcontext->__ss.__x[i]);
    }
    cd_code(uc->uc_mcontext->__ss.__pc);
    /* the JIT frame that called a C function: lr */
    cd_code(uc->uc_mcontext->__ss.__lr);
    /* and the caller of the crashing C function, through the frame chain */
    {
        uint64_t *fp = (uint64_t *)uc->uc_mcontext->__ss.__fp;
        for (int k = 0; k < 4 && fp; k++) {
            cd_str("\nframe ret "); cd_hex(fp[1]);
            if (k >= 1) cd_code(fp[1]);
            fp = (uint64_t *)fp[0];
        }
    }
    cd_str("\n");
    signal(sig, SIG_DFL);
    raise(sig);
}
static void cd_install(void)
{
    struct sigaction sa = { .sa_sigaction = cd_handler, .sa_flags = SA_SIGINFO };
    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGTRAP, &sa, NULL);
}

void ishjit_enable(void *envp)
{
    (void)envp;                 /* (routines set at the first translation) */
    /* the env offsets the hand-made encodings hold (scaled 12-bit, ldur) */
    QEMU_BUILD_BUG_ON(offsetof(CPUARMState, xregs[31]) / 8 > 4095 ||
                      offsetof(CPUARMState, xregs[0]) % 8);
    QEMU_BUILD_BUG_ON(offsetof(CPUARMState, pc) / 8 > 4095 ||
                      offsetof(CPUARMState, pc) % 8);
    QEMU_BUILD_BUG_ON(offsetof(CPUARMState, VF) / 4 > 4095 ||
                      offsetof(CPUARMState, NF) % 4 || offsetof(CPUARMState, ZF) % 4 ||
                      offsetof(CPUARMState, CF) % 4 || offsetof(CPUARMState, VF) % 4);
    QEMU_BUILD_BUG_ON(offsetof(CPUARMState, vfp.zregs[31]) / 16 > 4095 ||
                      offsetof(CPUARMState, vfp.zregs[0]) % 16);
    QEMU_BUILD_BUG_ON((int)offsetof(ArchCPU, parent_obj.neg.icount_decr.u32) -
                      (int)offsetof(ArchCPU, env) < -256);
    if (getenv("ISHJIT_CRASHDUMP")) {
        cd_install();
    }
    ishjit_gen_code_hook = ishjit_gen_code;
}
