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
#include "cpu.h"
#include "internals.h"
#include "exec/translation-block.h"
#include "tcg/tcg.h"
#include "tcg/insn-start-words.h"
#include "accel/tcg/ishjit.h"
#include "translate.h"
#include "tcg/tcg-ldst.h"
#include "exec/memopidx.h"

/* From accel/tcg (tcg-runtime.h): the TB to continue with, or the epilogue */
const void *helper_lookup_tb_ptr(CPUArchState *env);

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
    /* slow paths of the block's memory accesses */
    int nslow;
} IJ;

static inline void emit(IJ *j, uint32_t insn)
{
    *j->ptr++ = insn;
}

static inline intptr_t ij_offset(IJ *j)
{
    return (uint8_t *)j->ptr - (uint8_t *)j->start;
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
    } else {
        emit_movi(j, H_TMP1, v);
        emit(j, 0x8B000000 | H_TMP1 << 16 | rn << 5 | rd);  /* add rd, rn, x17 */
    }
}

static unsigned xreg_off(int g)
{
    return offsetof(CPUARMState, xregs[g]);
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
            if (!(insn & (1u << 31)) && (insn & (1u << 22))) {
                return false;                       /* N=1 with 32 bits */
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
                ((insn >> 31) & 1) != ((insn >> 22) & 1)) {
                return false;
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

/* Loads of the guest registers the block uses; NZCV if read first */
static void emit_block_entry(IJ *j)
{
    for (int g = 0; g < 32; g++) {
        if ((j->used >> g) & 1 && g2h[g] != H_NONE) {
            emit(j, i_ldr_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
    if (j->flags_in) {
        emit_flags_from_env(j);
    }
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
    for (int g = 0; g < 32; g++) {
        if ((j->written >> g) & 1 && g2h[g] != H_NONE) {
            emit(j, i_str_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
    if (j->flags_out) {
        emit_flags_to_env(j);
    }
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

/* env->pc += diff (TBs are CF_PCREL: the pc is up to date on entry) */
static void emit_pc_add(IJ *j, int64_t diff)
{
    if (diff) {
        emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
        emit_addi(j, H_TMP0, H_TMP0, diff);
        emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
    }
}

static void emit_exit_tb(IJ *j, unsigned idx)
{
    const void *ret = tcg_ishjit_tb_ret_addr();
    uintptr_t here = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    ptrdiff_t off;

    emit_movi(j, 0, (uintptr_t)tcg_splitwx_to_rx(j->tb) + idx);
    here = (uintptr_t)tcg_splitwx_to_rx(j->ptr);
    off = ((uintptr_t)ret - here) >> 2;
    if (off == sextract64(off, 0, 26)) {
        emit(j, 0x14000000 | (off & 0x3ffffff));
    } else {
        emit_movi(j, H_TMP0, (uintptr_t)ret);
        emit(j, i_br(H_TMP0));
    }
}

/* A chainable jump (registers already synced, pc updated) */
static void emit_goto_tb(IJ *j, int which)
{
    j->tb->jmp_insn_offset[which] = ij_offset(j);
    emit(j, 0x14000000);                    /* b . (patched) */
    emit(j, i_br(H_TMP0));                  /* (when patched to ldr x16) */
    j->tb->jmp_reset_offset[which] = ij_offset(j);
    emit_exit_tb(j, which);
}

/* To @dest from the insn at @pc_insn: synced, pc updated, then chained or looked up */
static void emit_jump(IJ *j, int which, vaddr dest)
{
    emit_pc_add(j, (int64_t)(dest - j->pc_first));
    if (((dest ^ j->pc_first) & TARGET_PAGE_MASK) == 0) {
        emit_goto_tb(j, which);
    } else {
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
enum { WB_NONE, WB_PRE, WB_POST };

typedef struct {
    uint8_t size;           /* log2 bytes per register */
    uint8_t opc;            /* bits [23:22] of the single-register forms */
    bool load, pair, sign, dest32;
    uint8_t rt, rt2, rn, rm;
    uint8_t mode, wb, option, shift;
    bool prefetch;
    int64_t imm;            /* byte offset (or the literal's pc offset) */
} MemInfo;

static bool decode_mem(uint32_t insn, MemInfo *m)
{
    memset(m, 0, sizeof(*m));
    m->rt = insn & 31;
    m->rn = (insn >> 5) & 31;
    if (m->rn == 31 && getenv("ISHJIT_NOSP") &&
        (insn & 0x3B000000) != 0x18000000) {
        return false;
    }

    if ((insn & 0x3B000000) == 0x39000000 ||        /* unsigned immediate */
        (insn & 0x3B200000) == 0x38000000 ||        /* imm9 forms */
        (insn & 0x3B200C00) == 0x38200800) {        /* register offset */
        unsigned size = insn >> 30, opc = (insn >> 22) & 3;

        if (insn & (1u << 26)) {
            return false;                           /* SIMD&FP (phase 3) */
        }
        if ((insn & 0x3B000000) == 0x39000000) {
            m->mode = M_IMM;
            m->imm = (int64_t)((insn >> 10) & 0xfff) << size;
        } else if ((insn & 0x3B200000) == 0x38000000) {
            unsigned kind = (insn >> 10) & 3;

            if (kind == 2) {
                return false;                       /* ldtr/sttr (unprivileged) */
            }
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
            return false;                           /* SIMD&FP (phase 3) */
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
            return false;
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
    if (m->rt != 31) {
        *(m->load ? writes : reads) |= 1u << m->rt;
    }
    if (m->pair && m->rt2 != 31) {
        *(m->load ? writes : reads) |= 1u << m->rt2;
    }
}

enum { K_STOP, K_DP, K_NOP, K_ADR, K_MEM, K_B, K_BL, K_BCOND, K_CB, K_TB, K_BREG };

static int classify(uint32_t insn, OperandInfo *o, MemInfo *m, bool mem_ok)
{
    if (mem_ok && decode_mem(insn, m)) {
        return m->prefetch ? K_NOP : K_MEM;
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
    bool flags;             /* NZCV valid in the host at this insn */
    bool allslow;           /* (debugging: always the slow path) */
} IJSlow;

/* Top-byte-ignore of data addresses (TCG's clean_data_tbi), on x16 */
static void emit_tbi(IJ *j)
{
    uint32_t sbfx = 0x93400000 | 55 << 10 | H_TMP0 << 5 | H_TMP0;  /* sbfx x16, x16, #0, #56 */

    switch (j->tbid) {
    case 3:
        emit(j, sbfx);
        break;
    case 1:     /* lower half only: bit 55 clear */
        emit(j, 0x37000000 | 1u << 31 | 23 << 19 | 2 << 5 | H_TMP0);  /* tbnz x16, #55, +8 */
        emit(j, sbfx);
        break;
    case 2:     /* upper half only */
        emit(j, 0x36000000 | 1u << 31 | 23 << 19 | 2 << 5 | H_TMP0);  /* tbz x16, #55, +8 */
        emit(j, sbfx);
        break;
    }
}

static int tlb_fast_off(int mmu_idx)
{
    return (int)offsetof(ArchCPU, parent_obj.neg.tlb.f[mmu_idx]) -
           (int)offsetof(ArchCPU, env);
}

/*
 * A guest integer load/store: the address in x16 (+ size - 1, so that the
 * TLB check also catches an access crossing a page), an inline check of
 * QEMU's softmmu TLB that leaves the flags alone, then the access itself on
 * the host address. Misses go to a slow path (after the block).
 */
static void emit_mem(IJ *j, const MemInfo *m, vaddr ipc, IJSlow *sl)
{
    int accsize = (1 << m->size) * (m->pair ? 2 : 1);
    unsigned pb = j->page_bits;
    int cmp_off = m->load ? offsetof(CPUTLBEntry, addr_read)
                          : offsetof(CPUTLBEntry, addr_write);

    if (m->mode == M_LIT) {
        emit(j, i_ldr_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
        emit_addi(j, H_TMP0, H_TMP0, (int64_t)(ipc + m->imm - j->pc_first));
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
        emit_tbi(j);
    }

    /*
     * The entry of the first byte's page (x27), compared (rotated: its flag
     * bits on top) with the page of the last byte: an access crossing pages
     * misses, as in TCG. x16 = the last byte from here on.
     */
    emit(j, 0xA9400000 | ((tlb_fast_off(j->mmu_idx) / 8) & 0x7f) << 15 |
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
    if (getenv("ISHJIT_ALLSLOW")) {
        sl->allslow = true;
    }
    emit(j, 0x8B000000 | H_TMP0 << 16 | H_TMP2 << 5 | H_TMP2);  /* add x27, x27, x16 */

    if (!m->pair) {
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
    for (int g = 0; g < 32; g++) {
        if ((j->used >> g) & 1 && g2h[g] != H_NONE) {
            emit(j, i_str_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
    emit(j, i_mov_x(1, H_TMP0));
    if (flags) {
        emit_flags_to_env(j);
    }
    emit(j, i_br(H_TMP2));
}

static void emit_restore_thunk(IJ *j, bool flags)
{
    if (flags) {
        emit_flags_from_env(j);
    }
    for (int g = 0; g < 32; g++) {
        if ((j->used >> g) & 1 && g2h[g] != H_NONE) {
            emit(j, i_ldr_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
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

static void emit_slow(IJ *j, const IJSlow *sl, uint32_t *save, uint32_t *restore)
{
    const MemInfo *m = &sl->m;
    int accsize = (1 << m->size) * (m->pair ? 2 : 1);
    MemOp mop = m->size | MO_LE | (m->sign ? MO_SIGN : 0);
    MemOpIdx oi = make_memop_idx(mop, j->mmu_idx);
    uint32_t *adr;

    *sl->cbnz = sl->allslow ? i_b(j->ptr - sl->cbnz) :
                0xB5000000 | ((j->ptr - sl->cbnz) & 0x7ffff) << 5 | H_TMP1;
    if (accsize > 1) {
        emit_addi(j, H_TMP0, H_TMP0, -(accsize - 1));
    }
    emit(j, i_adr(H_TMP2, 8));
    emit(j, i_b(save - j->ptr));
    if (m->pair) {
        emit(j, i_mov_x(20, 1));
    }
    for (int k = 0; k < (m->pair ? 2 : 1); k++) {
        int rt = k ? m->rt2 : m->rt;

        if (k) {
            emit_addi(j, 1, 20, 1 << m->size);
        }
        emit(j, i_mov_x(0, H_ENV));
        if (m->load) {
            emit_movi(j, 2, oi);
            adr = j->ptr;
            emit(j, i_adr(3, (sl->resume - adr) * 4));
            emit_movi(j, H_TMP0, (uintptr_t)ld_helper(m->size, m->sign));
            emit(j, i_blr(H_TMP0));
            if (m->dest32) {
                emit(j, 0x2A0003E0);                /* mov w0, w0 */
            }
            if (rt != 31) {
                emit(j, i_str_x(0, H_ENV, xreg_off(rt)));
            }
        } else {
            if (rt == 31) {
                emit(j, i_mov_x(2, 31));
            } else {
                emit(j, i_ldr_x(2, H_ENV, xreg_off(rt)));
            }
            emit_movi(j, 3, oi);
            adr = j->ptr;
            emit(j, i_adr(4, (sl->resume - adr) * 4));
            emit_movi(j, H_TMP0, (uintptr_t)st_helper(m->size));
            emit(j, i_blr(H_TMP0));
        }
    }
    adr = j->ptr;
    emit(j, i_adr(H_TMP2, (sl->resume - adr) * 4));
    emit(j, i_b(restore - j->ptr));
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
    IJSlow slows[TCG_MAX_INSNS];
    bool fwritten[TCG_MAX_INSNS];       /* NZCV written by an earlier insn */
    int kinds[TCG_MAX_INSNS];
    int n = 0, last;
    CPUARMTBFlags tbf = arm_tbflags_from_tb(tb);
    vaddr page_end = (pc & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
    uint64_t *data = tcg_ctx->gen_insn_data;
    uint16_t *end_off = tcg_ctx->gen_insn_end_off;

    /* Memory accesses: through the TB's MMU index, unless checks apply */
    j->mmu_idx = EX_TBFLAG_ANY(tbf, MMUIDX);
    j->tbid = EX_TBFLAG_A64(tbf, TBID);
    j->page_bits = TARGET_PAGE_BITS;
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
        int k = classify(insn, &ops[n], &mems[n], j->mem_ok);

        if (k == K_STOP) {
            break;
        }
        insns[n] = insn;
        kinds[n] = k;
        fwritten[n] = j->flags_written;
        switch (k) {
        case K_MEM:
            mem_regs(&mems[n], &rd, &wr);
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
        /* env-held registers: at most three per DP insn, none for cb/tb */
        {
            uint32_t mem = (rd | wr) & ENV_HELD;

            if ((k == K_DP && ctpop32(mem) > 3) ||
                ((k == K_CB || k == K_TB) && mem)) {
                break;
            }
        }
        j->used |= rd | wr;
        j->written |= wr;
        n++;
        if (k >= K_B) {
            break;                              /* a branch ends the block */
        }
    }
    if (getenv("ISHJIT_STATS")) {
        ishjit_stat(n, n < max_insns && pc + 4 * n < page_end && (n == 0 || kinds[n - 1] < K_B) ?
                    ldl_le_p((uint8_t *)host_pc + 4 * n) : 0, n == 0);
    }
    if (n == 0) {
        return ISHJIT_DECLINE;
    }
    last = n - 1;

    /* The TCG TB prologue's check: exit if an interrupt or exit is pending */
    {
        int off = (int)offsetof(ArchCPU, parent_obj.neg.icount_decr.u32) -
                  (int)offsetof(ArchCPU, env);
        uint32_t *tbnz;

        emit(j, i_ldur_w(H_TMP0, H_ENV, off));
        tbnz = j->ptr;
        emit(j, 0);                             /* tbnz w16, #31, exit */
        emit_block_entry(j);
        /* (the exit it jumps to is placed after the block's code) */
        tb->jmp_insn_offset[0] = TB_JMP_OFFSET_INVALID;
        tb->jmp_insn_offset[1] = TB_JMP_OFFSET_INVALID;
        tb->jmp_reset_offset[0] = TB_JMP_OFFSET_INVALID;
        tb->jmp_reset_offset[1] = TB_JMP_OFFSET_INVALID;

        for (int i = 0; i < n; i++) {
            uint32_t insn = insns[i];
            vaddr ipc = j->pc_first + 4 * i;

            if ((uint8_t *)j->ptr + 1024 > j->highwater) {
                return -1;                      /* code buffer full */
            }
            data[i * TARGET_INSN_START_WORDS + 0] = ipc & ~TARGET_PAGE_MASK;
            for (int w = 1; w < TARGET_INSN_START_WORDS; w++) {
                data[i * TARGET_INSN_START_WORDS + w] = 0;
            }

            switch (kinds[i]) {
            case K_NOP:
                break;
            case K_MEM:
                slows[j->nslow].m = mems[i];
                slows[j->nslow].allslow = false;
                slows[j->nslow].flags = j->flags_in || fwritten[i];
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
                /* the TB's guest pc is env->pc on entry */
                emit(j, i_ldr_x(h, H_ENV, offsetof(CPUARMState, pc)));
                if (insn & (1u << 31)) {        /* adrp */
                    val = (ipc & TARGET_PAGE_MASK) + (imm << 12);
                } else {
                    val = ipc + imm;
                }
                emit_addi(j, h, h, (int64_t)(val - j->pc_first));
                if (g2h[g] == H_NONE) {
                    emit(j, i_str_x(h, H_ENV, xreg_off(g)));
                }
                break;
            }
            case K_BL:
            case K_B: {
                vaddr dest = ipc + (sextract64(insn, 0, 26) << 2);

                if (kinds[i] == K_BL) {
                    emit(j, i_ldr_x(g2h[30], H_ENV, offsetof(CPUARMState, pc)));
                    emit_addi(j, g2h[30], g2h[30], (int64_t)(ipc + 4 - j->pc_first));
                }
                end_off[i] = ij_offset(j);
                emit_sync(j);
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
                emit_sync(j);
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
                    emit(j, i_ldr_x(g2h[30], H_ENV, offsetof(CPUARMState, pc)));
                    emit_addi(j, g2h[30], g2h[30], (int64_t)(ipc + 4 - j->pc_first));
                }
                end_off[i] = ij_offset(j);
                /* (before the sync, which uses the scratch registers) */
                emit(j, i_str_x(H_TMP0, H_ENV, offsetof(CPUARMState, pc)));
                emit_sync(j);
                emit(j, i_mov_x(0, H_ENV));
                emit_movi(j, H_TMP0, (uintptr_t)helper_lookup_tb_ptr);
                emit(j, i_blr(H_TMP0));
                emit(j, i_br(0));
                break;
            }
            }
            if (kinds[i] < K_B) {
                end_off[i] = ij_offset(j);
            }
        }

        /* A block not ended by a branch continues at the next insn */
        if (kinds[last] < K_B) {
            emit_sync(j);
            emit_jump(j, 0, j->pc_first + 4 * n);
        }

        /* Slow paths: the block's save/restore helpers, then one stub each */
        if (j->nslow) {
            uint32_t *save[2] = { NULL, NULL }, *restore[2] = { NULL, NULL };

            for (int f = 0; f < 2; f++) {
                bool need = false;

                for (int k = 0; k < j->nslow; k++) {
                    need |= slows[k].flags == f;
                }
                if (need) {
                    if ((uint8_t *)j->ptr + 1024 > j->highwater) {
                        return -1;
                    }
                    save[f] = j->ptr;
                    emit_save_thunk(j, f);
                    restore[f] = j->ptr;
                    emit_restore_thunk(j, f);
                }
            }
            for (int k = 0; k < j->nslow; k++) {
                if ((uint8_t *)j->ptr + 1024 > j->highwater) {
                    return -1;
                }
                emit_slow(j, &slows[k], save[slows[k].flags],
                          restore[slows[k].flags]);
            }
        }

        /* The pending-exit check's target: nothing is live yet */
        {
            ptrdiff_t d = j->ptr - tbnz;

            *tbnz = 0x37000000 | 31 << 19 | (d & 0x3fff) << 5 | H_TMP0;
            emit_exit_tb(j, TB_EXIT_REQUESTED);
        }
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

    if (!host_pc || !is_a64(env) ||
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

void ishjit_enable(void)
{
    ishjit_gen_code_hook = ishjit_gen_code;
}
