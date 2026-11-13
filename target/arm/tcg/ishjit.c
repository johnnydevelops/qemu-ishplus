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
 * Phase 1: integer data processing and branches. Memory accesses, FP/SIMD
 * and system instructions end the block (TCG translates from there).
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

/* From accel/tcg (tcg-runtime.h): the TB to continue with, or the epilogue */
const void *helper_lookup_tb_ptr(CPUArchState *env);

/*
 * Registers. Host x18 is Apple's (zeroed on context switches), host SP must
 * stay a host stack; x19 is env (TCG's TCG_AREG0, so TCG TBs and ishjit TBs
 * chain), x16/x17 are scratch. The 27 others hold guest registers: x0-x12,
 * x16, x17, x19-x27, x29, x30 and SP. The five least used guest registers
 * (measured over the Ubuntu workload: x13, x14, x15, x18, x28, together 1% of
 * register accesses) stay in env and are loaded into a scratch when used.
 */
#define H_ENV   19
#define H_TMP0  16
#define H_TMP1  17
#define H_NONE  (-1)

/* guest register (0-30, 31 = SP) -> host register, or H_NONE (in env) */
static const int8_t g2h[32] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    H_NONE, H_NONE, H_NONE,             /* x13-x15 */
    13, 14,                             /* x16, x17 */
    H_NONE,                             /* x18 */
    15,                                 /* x19 */
    20, 21, 22, 23, 24, 25, 26, 27,     /* x20-x27 */
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

/* Loads of the guest registers the block uses; NZCV if read first */
static void emit_block_entry(IJ *j)
{
    for (int g = 0; g < 32; g++) {
        if ((j->used >> g) & 1 && g2h[g] != H_NONE) {
            emit(j, i_ldr_x(g2h[g], H_ENV, xreg_off(g)));
        }
    }
    if (j->flags_in) {
        /* QEMU's NF/ZF/CF/VF to the host NZCV (flags are free here) */
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

enum { K_STOP, K_DP, K_NOP, K_ADR, K_B, K_BL, K_BCOND, K_CB, K_TB, K_BREG };

static int classify(uint32_t insn, OperandInfo *o)
{
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

static int ishjit_gen(CPUState *cs, TranslationBlock *tb, vaddr pc,
                      void *host_pc, int max_insns, void *buf)
{
    CPUARMState *env = cpu_env(cs);
    IJ jj = { .ptr = buf, .start = buf, .tb = tb, .env = env,
              .pc_first = pc, .highwater = tcg_ctx->code_gen_highwater };
    IJ *j = &jj;
    uint32_t insns[TCG_MAX_INSNS];
    OperandInfo ops[TCG_MAX_INSNS];
    int kinds[TCG_MAX_INSNS];
    int n = 0, last;
    vaddr page_end = (pc & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;
    uint64_t *data = tcg_ctx->gen_insn_data;
    uint16_t *end_off = tcg_ctx->gen_insn_end_off;

    /* Scan: the run of instructions ishjit translates, up to a branch */
    while (n < max_insns && pc + 4 * n < page_end) {
        uint32_t insn = ldl_le_p((uint8_t *)host_pc + 4 * n);
        uint32_t rd = 0, wr = 0;
        int k = classify(insn, &ops[n]);

        if (k == K_STOP) {
            break;
        }
        insns[n] = insn;
        kinds[n] = k;
        switch (k) {
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
        /* at most two of the env-held registers per insn (two scratches) */
        {
            uint32_t mem = (rd | wr) & ((1u << 13) | (1u << 14) | (1u << 15) |
                                        (1u << 18) | (1u << 28));
            if (ctpop32(mem) > 2 || (k == K_CB && mem) || (k == K_TB && mem)) {
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
            case K_DP: {
                const OperandInfo *o = &ops[i];
                int tmp[32];
                uint32_t rd, wr, mem;
                int next = H_TMP0;

                operand_regs(insn, o, &rd, &wr);
                mem = (rd | wr) & ((1u << 13) | (1u << 14) | (1u << 15) |
                                   (1u << 18) | (1u << 28));
                for (int g = 0; g < 31; g++) {
                    tmp[g] = H_NONE;
                    if ((mem >> g) & 1) {
                        tmp[g] = next;
                        next = H_TMP1;
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
    return ishjit_gen(cs, tb, pc, host_pc, max_insns, buf);
}

void ishjit_enable(void)
{
    ishjit_gen_code_hook = ishjit_gen_code;
}
