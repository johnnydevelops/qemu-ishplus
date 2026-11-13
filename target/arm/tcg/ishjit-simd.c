/*
 * ishjit: operand decoder for the A64 "Data Processing -- Scalar
 * Floating-Point and Advanced SIMD" encoding space (op0 = x111).
 *
 * Accepts exactly the instructions an Apple A13 (ARMv8.4-A) implements:
 * FP, AdvSIMD, FEAT_FP16, FEAT_FHM, FEAT_DotProd, FEAT_RDM, FEAT_JSCVT,
 * FEAT_FCMA, AES/PMULL, SHA1/SHA256/SHA512/SHA3.  Everything else (BF16,
 * I8MM, FRINTTS, SM3/SM4, FP8, FAMINMAX, LUT, unallocated) is rejected.
 * Verified against LLVM's disassembler by build/fastmem/simd/test/simd_test.py.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "qemu/osdep.h"
#include "ishjit-simd.h"

#define SF(pos, len)    ((insn >> (pos)) & ((1u << (len)) - 1))
#define SB(pos)         ((insn >> (pos)) & 1)
#define MSK(m, i)       (((m) >> (i)) & 1)

/*
 * FP opcode tables, indexed by U:a (bit 29, bit 23); bit n set means
 * opcode 0x18+n (5-bit opcode) or n (3-bit FP16 opcode) is allocated.
 */
static const uint8_t simd_3same_fp[4]    = { 0xff, 0xe7, 0xff, 0x77 };
static const uint8_t simd_3same_fp16[4]  = { 0xdf, 0xc7, 0xfd, 0x75 };
static const uint8_t simd_s3same_fp[4]   = { 0x98, 0x80, 0x30, 0x34 };
static const uint8_t simd_2misc_fp[4]    = { 0x3f, 0x3f, 0x3f, 0xbe };
static const uint8_t simd_2misc_fp16[4]  = { 0x3f, 0x2f, 0x3f, 0xae };
static const uint8_t simd_s2misc_fp[4]   = { 0x3c, 0xac, 0x3c, 0x2c };

/* AdvSIMD three same (vector and scalar) */
static bool simd_3same(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned q = SB(30), u = SB(29), size = SF(22, 2), op = SF(11, 5);

    if (op >= 0x18) {
        unsigned idx = (u << 1) | (size >> 1);

        g->reads_fpcr_mode = true;
        if (scalar) {
            return MSK(simd_s3same_fp[idx], op - 0x18);
        }
        if ((size & 1) && !q) {
            return false;
        }
        if (op == 0x1d && u == 0 && (size & 1)) {
            return false;                   /* FMLAL/FMLSL: sz must be 0 */
        }
        if (op == 0x19 && u == 1 && (size & 1)) {
            return false;                   /* FMLAL2/FMLSL2 */
        }
        return MSK(simd_3same_fp[idx], op - 0x18);
    }

    switch (op) {
    case 0x01: case 0x05: case 0x09: case 0x0b:     /* SQADD..UQRSHL */
        g->sets_qc = true;
        return scalar || size != 3 || q;
    case 0x06: case 0x07: case 0x08: case 0x0a: case 0x10: case 0x11:
        return scalar ? size == 3 : (size != 3 || q);
    case 0x16:                                      /* SQDMULH/SQRDMULH */
        g->sets_qc = true;
        return size == 1 || size == 2;
    }
    if (scalar) {
        return false;
    }
    switch (op) {
    case 0x00: case 0x02: case 0x04: case 0x0c: case 0x0d: case 0x0e:
    case 0x0f: case 0x12: case 0x14: case 0x15:
        return size != 3;
    case 0x03:                                      /* logical */
        return true;
    case 0x13:                                      /* MUL / PMUL */
        return u ? size == 0 : size != 3;
    case 0x17:                                      /* ADDP */
        return !u && (size != 3 || q);
    }
    return false;
}

/* AdvSIMD three same (FP16), vector and scalar */
static bool simd_3same_h(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned idx = (SB(29) << 1) | SB(23), op = SF(11, 3);

    g->reads_fpcr_mode = true;
    if (scalar) {
        return MSK(simd_s3same_fp[idx], op);
    }
    return MSK(simd_3same_fp16[idx], op);
}

/* AdvSIMD three same extra, vector and scalar */
static bool simd_3same_extra(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned q = SB(30), u = SB(29), size = SF(22, 2), op = SF(11, 4);

    if (!u) {
        return !scalar && op == 2 && size == 2;     /* SDOT */
    }
    if (op <= 1) {                                  /* SQRDMLAH/SQRDMLSH */
        g->sets_qc = true;
        return size == 1 || size == 2;
    }
    if (scalar) {
        return false;
    }
    if (op == 2) {                                  /* UDOT */
        return size == 2;
    }
    if ((op & 0xc) == 0x8 || op == 0xc || op == 0xe) {  /* FCMLA, FCADD */
        g->reads_fpcr_mode = true;
        return size == 1 || size == 2 || (size == 3 && q);
    }
    return false;
}

/* AdvSIMD three different, vector and scalar */
static bool simd_3diff(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned u = SB(29), size = SF(22, 2), op = SF(12, 4);

    if (op == 9 || op == 11 || op == 13) {          /* SQDMLAL/SQDMLSL/SQDMULL */
        g->sets_qc = true;
        return !u && (size == 1 || size == 2);
    }
    if (scalar) {
        return false;
    }
    if (op == 14) {                                 /* PMULL */
        return !u && (size == 0 || size == 3);
    }
    return op != 15 && size != 3;
}

/* AdvSIMD two-register miscellaneous, vector */
static bool simd_2misc(uint32_t insn, SimdOperands *g)
{
    unsigned q = SB(30), u = SB(29), size = SF(22, 2), op = SF(12, 5);
    unsigned a = size >> 1, sz = size & 1;

    switch (op) {
    case 0x00:
        return u ? size < 2 : size != 3;            /* REV32 / REV64 */
    case 0x01:
        return !u && size == 0;                     /* REV16 */
    case 0x02: case 0x04: case 0x06:                /* xADDLP, CLS/CLZ, xADALP */
        return size != 3;
    case 0x05:
        return u ? size < 2 : size == 0;            /* NOT,RBIT / CNT */
    case 0x03: case 0x07:                           /* xQADD, SQABS/SQNEG */
        g->sets_qc = true;
        return size != 3 || q;
    case 0x08: case 0x09: case 0x0b:
        return size != 3 || q;
    case 0x0a:
        return !u && (size != 3 || q);              /* CMLT #0 */
    case 0x12:                                      /* XTN / SQXTUN */
        g->sets_qc = u;
        return size != 3;
    case 0x13:                                      /* SHLL */
        return u && size != 3;
    case 0x14:                                      /* SQXTN / UQXTN */
        g->sets_qc = true;
        return size != 3;
    case 0x16:                                      /* FCVTN / FCVTXN */
        g->reads_fpcr_mode = true;
        return !a && (!u || sz);
    case 0x17:                                      /* FCVTL */
        g->reads_fpcr_mode = true;
        return !u && !a;
    }
    if (sz && !q) {
        return false;
    }
    if (op >= 0x0c && op <= 0x0f) {
        if (!a || (u && op == 0x0e)) {
            return false;
        }
        g->reads_fpcr_mode = op != 0x0f;            /* not FABS/FNEG */
        return true;
    }
    if (op >= 0x18) {
        if (op == 0x1c && a) {                      /* URECPE/URSQRTE */
            return !sz;
        }
        g->reads_fpcr_mode = true;
        return MSK(simd_2misc_fp[(u << 1) | a], op - 0x18);
    }
    return false;
}

/* AdvSIMD two-register miscellaneous (FP16), vector and scalar */
static bool simd_2misc_h(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned u = SB(29), a = SB(23), op = SF(12, 5);

    g->reads_fpcr_mode = true;
    if (op >= 0x0c && op <= 0x0f) {
        if (!a || (u && op == 0x0e)) {
            return false;
        }
        if (op == 0x0f) {
            g->reads_fpcr_mode = false;             /* FABS/FNEG */
            return !scalar;
        }
        return true;
    }
    if (op >= 0x18) {
        if (scalar) {
            return MSK(simd_s2misc_fp[(u << 1) | a], op - 0x18);
        }
        return MSK(simd_2misc_fp16[(u << 1) | a], op - 0x18);
    }
    return false;
}

/* AdvSIMD scalar two-register miscellaneous */
static bool simd_s2misc(uint32_t insn, SimdOperands *g)
{
    unsigned u = SB(29), size = SF(22, 2), op = SF(12, 5);
    unsigned a = size >> 1, sz = size & 1;

    switch (op) {
    case 0x03: case 0x07:                           /* xQADD, SQABS/SQNEG */
        g->sets_qc = true;
        return true;
    case 0x08: case 0x09: case 0x0b:
        return size == 3;
    case 0x0a:
        return !u && size == 3;
    case 0x12:
        g->sets_qc = true;
        return u && size != 3;                      /* SQXTUN */
    case 0x14:
        g->sets_qc = true;
        return size != 3;                           /* SQXTN / UQXTN */
    case 0x16:
        g->reads_fpcr_mode = true;
        return u && !a && sz;                       /* FCVTXN */
    }
    g->reads_fpcr_mode = true;
    if (op >= 0x0c && op <= 0x0e) {
        return a && !(u && op == 0x0e);
    }
    if (op >= 0x18) {
        return MSK(simd_s2misc_fp[(u << 1) | a], op - 0x18);
    }
    return false;
}

/* AdvSIMD across lanes */
static bool simd_across(uint32_t insn, SimdOperands *g)
{
    unsigned q = SB(30), u = SB(29), size = SF(22, 2), op = SF(12, 5);

    switch (op) {
    case 0x03: case 0x0a: case 0x1a:                /* xADDLV, xMAXV, xMINV */
        return size != 3 && !(size == 2 && !q);
    case 0x1b:                                      /* ADDV */
        return !u && size != 3 && !(size == 2 && !q);
    case 0x0c: case 0x0f:                           /* FMAXNMV.. FMINV */
        g->reads_fpcr_mode = true;
        if (size & 1) {
            return false;
        }
        return !u || q;                             /* U=0 is FP16 */
    }
    return false;
}

/* AdvSIMD scalar pairwise */
static bool simd_spair(uint32_t insn, SimdOperands *g)
{
    unsigned u = SB(29), size = SF(22, 2), op = SF(12, 5);

    if (op == 0x1b) {
        return !u && size == 3;                     /* ADDP */
    }
    g->reads_fpcr_mode = true;
    if (!u && (size & 1)) {
        return false;                               /* U=0 is FP16 */
    }
    return op == 0x0c || op == 0x0f || (op == 0x0d && !(size >> 1));
}

/* AdvSIMD copy, vector and scalar */
static bool simd_copy(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned q = SB(30), op = SB(29), imm5 = SF(16, 5), imm4 = SF(11, 4);
    unsigned size;

    if (!(imm5 & 0xf)) {
        return false;
    }
    size = imm5 & 1 ? 0 : imm5 & 2 ? 1 : imm5 & 4 ? 2 : 3;
    if (scalar) {
        return !op && imm4 == 0;                    /* DUP (scalar) */
    }
    if (op) {
        return q;                                   /* INS (element) */
    }
    switch (imm4) {
    case 0:                                         /* DUP (element) */
        return size != 3 || q;
    case 1:                                         /* DUP (general) */
        g->rn = SIMD_GPR_ZR;
        return size != 3 || q;
    case 3:                                         /* INS (general) */
        g->rn = SIMD_GPR_ZR;
        return q;
    case 5:                                         /* SMOV */
        g->rd = SIMD_GPR_ZR;
        g->rd_written = true;
        return size < 2 + q;
    case 7:                                         /* UMOV */
        g->rd = SIMD_GPR_ZR;
        g->rd_written = true;
        return q ? size == 3 : size < 3;
    }
    return false;
}

/* AdvSIMD vector x indexed element, vector and scalar */
static bool simd_indexed(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned q = SB(30), u = SB(29), size = SF(22, 2), op = SF(12, 4);
    unsigned l = SB(21), h = SB(11);
    bool fp = false, qc = false, ok_int;

    switch (op | (u << 4)) {
    case 0x01: case 0x05: case 0x09: case 0x19:     /* FMLA FMLS FMUL FMULX */
        fp = true;
        break;
    case 0x03: case 0x07: case 0x0b: case 0x0c: case 0x0d:
    case 0x1d: case 0x1f:                           /* SQDML*, SQ(R)DMULH, RDM */
        qc = true;
        break;
    case 0x02: case 0x06: case 0x08: case 0x0a:     /* SMLAL SMLSL MUL SMULL */
    case 0x10: case 0x12: case 0x14: case 0x16: case 0x1a:
        if (scalar) {
            return false;
        }
        break;
    case 0x0e: case 0x1e:                           /* SDOT / UDOT */
        return !scalar && size == 2;
    case 0x00: case 0x04: case 0x18: case 0x1c:     /* FMLAL.. FMLSL2 */
        g->reads_fpcr_mode = true;
        return !scalar && size == 2;
    case 0x11: case 0x13: case 0x15: case 0x17:     /* FCMLA */
        g->reads_fpcr_mode = true;
        if (scalar) {
            return false;
        }
        if (size == 1) {
            return q || !h;
        }
        return size == 2 && q && !l;
    default:
        return false;
    }

    if (fp) {
        g->reads_fpcr_mode = true;
        if (size == 3) {
            return !l && (q || scalar);
        }
        return size != 1;
    }
    ok_int = size == 1 || size == 2;
    g->sets_qc = qc;
    return ok_int;
}

/* AdvSIMD modified immediate */
static bool simd_modimm(uint32_t insn)
{
    unsigned q = SB(30), op = SB(29), cmode = SF(12, 4), o2 = SB(11);

    if (o2) {
        return !op && cmode == 0xf;                 /* FMOV (half) */
    }
    if (op && cmode == 0xf) {
        return q;                                   /* FMOV .2D */
    }
    return true;
}

/* AdvSIMD shift by immediate, vector and scalar */
static bool simd_shift(uint32_t insn, SimdOperands *g, bool scalar)
{
    unsigned q = SB(30), u = SB(29), immh = SF(19, 4), op = SF(11, 5);
    bool d = immh & 8;

    if (scalar) {
        q = 1;
    }
    if (d && !q) {
        return false;
    }
    switch (op) {
    case 0x00: case 0x02: case 0x04: case 0x06: case 0x0a:
        return !scalar || d;
    case 0x08:                                      /* SRI */
        return u && (!scalar || d);
    case 0x0c:                                      /* SQSHLU */
        g->sets_qc = true;
        return u;
    case 0x0e:                                      /* SQSHL / UQSHL */
        g->sets_qc = true;
        return true;
    case 0x10: case 0x11:                           /* (R)SHRN, SQ(R)SHRUN */
        g->sets_qc = u;
        return !d && (u || !scalar);
    case 0x12: case 0x13:                           /* SQ(R)SHRN, UQ(R)SHRN */
        g->sets_qc = true;
        return !d;
    case 0x14:                                      /* SSHLL / USHLL */
        return !d && !scalar;
    case 0x1c: case 0x1f:                           /* xCVTF, FCVTZx fixed */
        g->reads_fpcr_mode = true;
        return immh > 1;
    }
    return false;
}

/* Cryptographic and SHA3 (0xce...) */
static bool simd_crypto_ce(uint32_t insn)
{
    if ((insn & 0xff808000) == 0xce000000) {        /* EOR3, BCAX */
        return SF(21, 2) <= 1;
    }
    if ((insn & 0xffe00000) == 0xce800000) {        /* XAR */
        return true;
    }
    if ((insn & 0xffe0b000) == 0xce608000) {        /* SHA512H.., RAX1 */
        return !SB(14);
    }
    if ((insn & 0xfffff000) == 0xcec08000) {        /* SHA512SU0 */
        return SF(10, 2) == 0;
    }
    return false;
}

/* Floating-point data-processing (bit 30 = 0, bit 28 = 1) */
static bool simd_fp(uint32_t insn, SimdOperands *g)
{
    unsigned sf = SB(31), ptype = SF(22, 2);
    unsigned rmode = SF(19, 2), op = SF(16, 3);

    if (SB(29)) {
        return false;
    }
    if (SB(24)) {                                   /* 3-source */
        g->reads_fpcr_mode = true;
        return !sf && ptype != 2;
    }
    if (!SB(21)) {                                  /* <-> fixed-point */
        if (ptype == 2 || (!sf && !SB(15))) {
            return false;
        }
        g->reads_fpcr_mode = true;
        if (rmode == 0 && (op == 2 || op == 3)) {   /* SCVTF/UCVTF */
            g->rn = SIMD_GPR_ZR;
            return true;
        }
        if (rmode == 3 && op <= 1) {                /* FCVTZS/FCVTZU */
            g->rd = SIMD_GPR_ZR;
            g->rd_written = true;
            return true;
        }
        return false;
    }
    if (SF(10, 2) != 0) {
        if (sf || ptype == 2) {
            return false;
        }
        switch (SF(10, 2)) {
        case 1:                                     /* FCCMP(E) */
            g->reads_nzcv = g->writes_nzcv = true;
            g->reads_fpcr_mode = true;
            return true;
        case 2:                                     /* 2-source */
            g->reads_fpcr_mode = true;
            return SF(12, 4) <= 8;
        default:                                    /* FCSEL */
            g->reads_nzcv = true;
            return true;
        }
    }
    if (SB(12)) {                                   /* FMOV (scalar, imm) */
        return !sf && ptype != 2 && SF(5, 5) == 0;
    }
    if (SB(13)) {                                   /* FCMP(E) */
        g->writes_nzcv = true;
        g->reads_fpcr_mode = true;
        if (SB(3) && SF(16, 5) != 0) {
            return false;                           /* #0.0 form: Rm SBZ */
        }
        return !sf && ptype != 2 && SF(14, 2) == 0 && SF(0, 3) == 0;
    }
    if (SB(14)) {                                   /* 1-source */
        unsigned opc = SF(15, 6);

        if (sf || ptype == 2) {
            return false;
        }
        g->reads_fpcr_mode = opc > 2;
        switch (opc) {
        case 0x00: case 0x01: case 0x02: case 0x03: /* FMOV FABS FNEG FSQRT */
        case 0x08: case 0x09: case 0x0a: case 0x0b: /* FRINTN/P/M/Z */
        case 0x0c: case 0x0e: case 0x0f:            /* FRINTA/X/I */
            return true;
        case 0x04:
            return ptype != 0;                      /* FCVT to single */
        case 0x05:
            return ptype != 1;                      /* FCVT to double */
        case 0x07:
            return ptype != 3;                      /* FCVT to half */
        }
        return false;
    }
    if (SB(15)) {
        return false;
    }

    /* <-> integer */
    if (ptype == 2) {
        if (!(sf && rmode == 1 && op >= 6)) {
            return false;
        }
    } else if (op >= 2 && rmode != 0 && !(op == 6 && rmode == 3)) {
        return false;
    }
    switch (op) {
    case 0: case 1: case 4: case 5:                 /* FCVTxx to GPR */
        g->reads_fpcr_mode = true;
        g->rd = SIMD_GPR_ZR;
        g->rd_written = true;
        return true;
    case 2: case 3:                                 /* SCVTF / UCVTF */
        g->reads_fpcr_mode = true;
        g->rn = SIMD_GPR_ZR;
        return true;
    case 6:
        g->rd = SIMD_GPR_ZR;
        g->rd_written = true;
        if (rmode == 3) {                           /* FJCVTZS */
            g->reads_fpcr_mode = true;
            g->writes_nzcv = true;
            return !sf && ptype == 1;
        }
        break;
    default:
        g->rn = SIMD_GPR_ZR;
        break;
    }
    /* FMOV (general) */
    if (ptype == 2) {
        return true;
    }
    return ptype == 3 || ptype == sf;
}

bool ishjit_simd_decode(uint32_t insn, SimdOperands *g)
{
    bool ok;

    memset(g, 0, sizeof(*g));
    if ((insn & 0x0e000000) != 0x0e000000) {
        return false;
    }
    if ((insn & 0x50000000) == 0x10000000) {
        ok = simd_fp(insn, g);
    } else if ((insn & 0xff000000) == 0xce000000) {
        ok = simd_crypto_ce(insn);
    } else if (insn & 0x80000000) {
        ok = false;
    } else {
        bool scalar = SB(28);

        if (SB(24)) {
            if (!SB(10)) {
                ok = simd_indexed(insn, g, scalar);
            } else if (SB(23)) {
                ok = false;
            } else if (SF(19, 4) == 0) {
                ok = !scalar && simd_modimm(insn);
            } else {
                ok = simd_shift(insn, g, scalar);
            }
        } else if (SB(21)) {
            switch (SF(10, 2)) {
            case 0:
                ok = simd_3diff(insn, g, scalar);
                break;
            case 2:
                switch (SF(17, 4)) {
                case 0x0:
                    ok = scalar ? simd_s2misc(insn, g) : simd_2misc(insn, g);
                    break;
                case 0x8:
                    ok = scalar ? simd_spair(insn, g) : simd_across(insn, g);
                    break;
                case 0xc:
                    ok = SB(22) && simd_2misc_h(insn, g, scalar);
                    break;
                case 0x4:
                    /* AES (vector, Q=1 U=0) and SHA two-reg (scalar U=0) */
                    if ((insn & 0xff3e0c00) == 0x4e280800) {
                        ok = SF(22, 2) == 0 && (SF(12, 5) & 0x1c) == 0x04;
                    } else if ((insn & 0xff3e0c00) == 0x5e280800) {
                        ok = SF(22, 2) == 0 && SF(12, 5) <= 2;
                    } else {
                        ok = false;
                    }
                    break;
                default:
                    ok = false;
                    break;
                }
                break;
            default:
                ok = simd_3same(insn, g, scalar);
                break;
            }
        } else if (SB(10)) {
            if (SB(15)) {
                ok = simd_3same_extra(insn, g, scalar);
            } else if (SF(21, 3) == 0) {
                ok = simd_copy(insn, g, scalar);
            } else if (SF(21, 2) == 2 && SF(14, 2) == 0) {
                ok = simd_3same_h(insn, g, scalar);
            } else {
                ok = false;
            }
        } else if (scalar) {
            /* SHA three-reg: 0101 1110 000 Rm 0 opc 00 */
            ok = (insn & 0xffe08c00) == 0x5e000000 && SF(12, 3) != 7;
        } else if (SB(15)) {
            ok = false;
        } else if (SB(29)) {
            /* EXT */
            ok = SF(22, 2) == 0 && (SB(30) || !SB(14));
        } else if (SF(22, 2) == 0 && SF(10, 2) == 0) {
            ok = true;                              /* TBL/TBX */
        } else if (SF(10, 2) == 2) {
            ok = (SF(12, 3) & 3) != 0 && (SF(22, 2) != 3 || SB(30));
        } else {
            ok = false;
        }
    }
    if (!ok) {
        memset(g, 0, sizeof(*g));
    }
    return ok;
}

#undef SF
#undef SB
#undef MSK
