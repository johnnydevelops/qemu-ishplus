/*
 * iSH+: AArch64 floating point on an AArch64 host's own FPU.
 *
 * QEMU computes guest floating point in software (softfloat), so that the
 * results and exception flags are exact on any host. On an AArch64 host the
 * FPU follows the same architecture: with the same FPCR it gives the
 * guest's results, NaNs and flags bit for bit. So for float statuses with
 * the host's default FPCR behaviour (FPCR.AH = 0, round to nearest even,
 * no FZ or DN: how Linux runs programs), the hot helpers run the host
 * instruction instead.
 *
 * Exception flags: the host's cumulative FPSR flags are the guest's. They
 * collect in the vCPU thread's FPSR and are folded into the A64 float
 * status when the guest reads FPSR (vfp_get_fpsr_from_host), cleared when
 * it writes FPSR, and folded in whenever the vCPU leaves cpu_exec (so other
 * threads, e.g. saving the VM, see them).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#ifndef TARGET_ARM_NATIVE_FP_H
#define TARGET_ARM_NATIVE_FP_H

#include "fpu/softfloat.h"

#if defined(__aarch64__) && !defined(CONFIG_ARM_NO_NATIVE_FP)
#define ARM_NATIVE_FP 1
#include <arm_neon.h>

/*
 * Whether operations under @s can run natively: ARM's ordinary behaviour,
 * round to nearest even, no flush to zero and no default NaN, which is how
 * Linux runs programs and the host FPCR's own setting (never changed here,
 * so it is never read either: reading FPCR on every operation cost more
 * than it saved). Other modes use softfloat.
 */
static inline bool arm_native_fp_ok(float_status *s)
{
    /*
     * Everything but the exception flags must be ARM's defaults with
     * FPCR = 0 (arm_set_default_fp_behaviours on a zeroed status): one
     * compare of the bytes after the flags, cheaper than field by field.
     */
    static const float_status ref = {
        .float_rounding_mode = float_round_nearest_even,
        .float_2nan_prop_rule = float_2nan_prop_s_ab,
        .float_3nan_prop_rule = float_3nan_prop_s_cab,
        .float_infzeronan_rule = float_infzeronan_dnan_if_qnan,
        .tininess_before_rounding = true,
        .ftz_detection = float_ftz_before_rounding,
        .default_nan_pattern = 0b01000000,
    };
    const size_t start = offsetof(float_status, float_rounding_mode);

    return likely(memcmp((const char *)s + start, (const char *)&ref + start,
                         sizeof(float_status) - start) == 0);
}

/* The host FPSR's cumulative flags as softfloat flags, and clears them */
static inline int arm_native_fp_take_flags(void)
{
    uint64_t fpsr;
    int flags = 0;

    asm volatile("mrs %0, fpsr" : "=r"(fpsr) : : "memory");
    if (likely(!(fpsr & 0x9f))) {
        return 0;
    }
    asm volatile("msr fpsr, %0" : : "r"(fpsr & ~(uint64_t)0x9f) : "memory");
    if (fpsr & (1 << 0)) {
        flags |= float_flag_invalid;
    }
    if (fpsr & (1 << 1)) {
        flags |= float_flag_divbyzero;
    }
    if (fpsr & (1 << 2)) {
        flags |= float_flag_overflow;
    }
    if (fpsr & (1 << 3)) {
        flags |= float_flag_underflow;
    }
    if (fpsr & (1 << 4)) {
        flags |= float_flag_inexact;
    }
    if (fpsr & (1 << 7)) {
        flags |= float_flag_input_denormal_flushed;
    }
    return flags;
}

static inline void arm_native_fp_clear_flags(void)
{
    uint64_t fpsr;

    asm volatile("mrs %0, fpsr" : "=r"(fpsr) : : "memory");
    asm volatile("msr fpsr, %0" : : "r"(fpsr & ~(uint64_t)0x9f) : "memory");
}

/*
 * Scalar operations, as single instructions (asm volatile: not moved before
 * the FPCR is set, nor folded by the compiler).
 */
#define NATIVE_BINOP(NAME, INSN, TYPE, REG, CT)                             \
static inline TYPE native_##NAME(TYPE a, TYPE b)                            \
{                                                                           \
    CT x, y, r;                                                             \
    TYPE out;                                                               \
    memcpy(&x, &a, sizeof(x));                                              \
    memcpy(&y, &b, sizeof(y));                                              \
    asm volatile(INSN " %" REG "0, %" REG "1, %" REG "2"                    \
                 : "=w"(r) : "w"(x), "w"(y));                               \
    memcpy(&out, &r, sizeof(out));                                          \
    return out;                                                             \
}

NATIVE_BINOP(adds, "fadd", float32, "s", float)
NATIVE_BINOP(subs, "fsub", float32, "s", float)
NATIVE_BINOP(muls, "fmul", float32, "s", float)
NATIVE_BINOP(divs, "fdiv", float32, "s", float)
NATIVE_BINOP(addd, "fadd", float64, "d", double)
NATIVE_BINOP(subd, "fsub", float64, "d", double)
NATIVE_BINOP(muld, "fmul", float64, "d", double)
NATIVE_BINOP(divd, "fdiv", float64, "d", double)
/* min/max, absolute difference and compares (masks), FPCR.AH = 0 forms */
NATIVE_BINOP(maxs, "fmax", float32, "s", float)
NATIVE_BINOP(mins, "fmin", float32, "s", float)
NATIVE_BINOP(maxnums, "fmaxnm", float32, "s", float)
NATIVE_BINOP(minnums, "fminnm", float32, "s", float)
NATIVE_BINOP(maxd, "fmax", float64, "d", double)
NATIVE_BINOP(mind, "fmin", float64, "d", double)
NATIVE_BINOP(maxnumd, "fmaxnm", float64, "d", double)
NATIVE_BINOP(minnumd, "fminnm", float64, "d", double)
NATIVE_BINOP(abds, "fabd", float32, "s", float)
NATIVE_BINOP(abdd, "fabd", float64, "d", double)
NATIVE_BINOP(ceqs, "fcmeq", float32, "s", float)
NATIVE_BINOP(cges, "fcmge", float32, "s", float)
NATIVE_BINOP(cgts, "fcmgt", float32, "s", float)
NATIVE_BINOP(acges, "facge", float32, "s", float)
NATIVE_BINOP(acgts, "facgt", float32, "s", float)
NATIVE_BINOP(ceqd, "fcmeq", float64, "d", double)
NATIVE_BINOP(cged, "fcmge", float64, "d", double)
NATIVE_BINOP(cgtd, "fcmgt", float64, "d", double)
NATIVE_BINOP(acged, "facge", float64, "d", double)
NATIVE_BINOP(acgtd, "facgt", float64, "d", double)
#undef NATIVE_BINOP

#define NATIVE_SQRT(NAME, TYPE, REG, CT)                                    \
static inline TYPE native_##NAME(TYPE a)                                    \
{                                                                           \
    CT x, r;                                                                \
    TYPE out;                                                               \
    memcpy(&x, &a, sizeof(x));                                              \
    asm volatile("fsqrt %" REG "0, %" REG "1" : "=w"(r) : "w"(x));          \
    memcpy(&out, &r, sizeof(out));                                          \
    return out;                                                             \
}

NATIVE_SQRT(sqrts, float32, "s", float)
NATIVE_SQRT(sqrtd, float64, "d", double)
#undef NATIVE_SQRT

/* a * b + c, fused */
#define NATIVE_FMA(NAME, TYPE, REG, CT)                                     \
static inline TYPE native_##NAME(TYPE a, TYPE b, TYPE c)                    \
{                                                                           \
    CT x, y, z, r;                                                          \
    TYPE out;                                                               \
    memcpy(&x, &a, sizeof(x));                                              \
    memcpy(&y, &b, sizeof(y));                                              \
    memcpy(&z, &c, sizeof(z));                                              \
    asm volatile("fmadd %" REG "0, %" REG "1, %" REG "2, %" REG "3"         \
                 : "=w"(r) : "w"(x), "w"(y), "w"(z));                       \
    memcpy(&out, &r, sizeof(out));                                          \
    return out;                                                             \
}

NATIVE_FMA(fmas, float32, "s", float)
NATIVE_FMA(fmad, float64, "d", double)
#undef NATIVE_FMA

/*
 * As arm_native_fp_ok(), but any rounding mode: for float to integer
 * conversions, whose host instruction names the rounding.
 */
static inline bool arm_native_fp_ok_anyround(float_status *s)
{
    float_status t = *s;

    t.float_rounding_mode = float_round_nearest_even;
    return arm_native_fp_ok(&t);
}

/* FCMP/FCMPE: NZCV as PSTATE bits */
#define NATIVE_FCMP(NAME, INSN, TYPE, REG, CT)                              \
static inline uint64_t native_##NAME(TYPE a, TYPE b)                        \
{                                                                           \
    CT x, y;                                                                \
    uint64_t nzcv;                                                          \
    memcpy(&x, &a, sizeof(x));                                              \
    memcpy(&y, &b, sizeof(y));                                              \
    asm volatile(INSN " %" REG "1, %" REG "2\n mrs %0, nzcv"                \
                 : "=r"(nzcv) : "w"(x), "w"(y) : "cc");                     \
    return nzcv;                                                            \
}
NATIVE_FCMP(cmps, "fcmp", float32, "s", float)
NATIVE_FCMP(cmpes, "fcmpe", float32, "s", float)
NATIVE_FCMP(cmpd, "fcmp", float64, "d", double)
NATIVE_FCMP(cmped, "fcmpe", float64, "d", double)
#undef NATIVE_FCMP

static inline float64 native_fcvtds(float32 a)
{
    float x; double r; float64 out;
    memcpy(&x, &a, 4);
    asm volatile("fcvt %d0, %s1" : "=w"(r) : "w"(x));
    memcpy(&out, &r, 8);
    return out;
}

static inline float32 native_fcvtsd(float64 a)
{
    double x; float r; float32 out;
    memcpy(&x, &a, 8);
    asm volatile("fcvt %s0, %d1" : "=w"(r) : "w"(x));
    memcpy(&out, &r, 4);
    return out;
}

/* Integer to float (the FPCR's rounding: nearest even here) */
#define NATIVE_ITOF(NAME, INSN, ITYPE, IREG, FTYPE, FREG, CT)                \
static inline FTYPE native_##NAME(ITYPE a)                                  \
{                                                                           \
    CT r; FTYPE out;                                                        \
    asm volatile(INSN " %" FREG "0, %" IREG "1" : "=w"(r) : "r"(a));         \
    memcpy(&out, &r, sizeof(out));                                          \
    return out;                                                             \
}
NATIVE_ITOF(sltod, "scvtf", int32_t, "w", float64, "d", double)
NATIVE_ITOF(ultod, "ucvtf", uint32_t, "w", float64, "d", double)
NATIVE_ITOF(sqtod, "scvtf", int64_t, "x", float64, "d", double)
NATIVE_ITOF(uqtod, "ucvtf", uint64_t, "x", float64, "d", double)
NATIVE_ITOF(sltos, "scvtf", int32_t, "w", float32, "s", float)
NATIVE_ITOF(ultos, "ucvtf", uint32_t, "w", float32, "s", float)
NATIVE_ITOF(sqtos, "scvtf", int64_t, "x", float32, "s", float)
NATIVE_ITOF(uqtos, "ucvtf", uint64_t, "x", float32, "s", float)
#undef NATIVE_ITOF

/*
 * Float to integer with a rounding mode: the instruction for it, or false
 * (then softfloat) for modes without one.
 */
#define NATIVE_FTOI(NAME, SIGN, ITYPE, IREG, FTYPE, FREG, CT)                \
static inline bool native_##NAME(FTYPE a, FloatRoundMode rmode, ITYPE *out)  \
{                                                                           \
    CT x; ITYPE r;                                                          \
    memcpy(&x, &a, sizeof(x));                                              \
    switch (rmode) {                                                        \
    case float_round_nearest_even:                                          \
        asm volatile("fcvtn" SIGN " %" IREG "0, %" FREG "1" : "=r"(r) : "w"(x)); \
        break;                                                              \
    case float_round_to_zero:                                               \
        asm volatile("fcvtz" SIGN " %" IREG "0, %" FREG "1" : "=r"(r) : "w"(x)); \
        break;                                                              \
    case float_round_up:                                                    \
        asm volatile("fcvtp" SIGN " %" IREG "0, %" FREG "1" : "=r"(r) : "w"(x)); \
        break;                                                              \
    case float_round_down:                                                  \
        asm volatile("fcvtm" SIGN " %" IREG "0, %" FREG "1" : "=r"(r) : "w"(x)); \
        break;                                                              \
    case float_round_ties_away:                                             \
        asm volatile("fcvta" SIGN " %" IREG "0, %" FREG "1" : "=r"(r) : "w"(x)); \
        break;                                                              \
    default:                                                                \
        return false;                                                       \
    }                                                                       \
    *out = r;                                                               \
    return true;                                                            \
}
NATIVE_FTOI(tosld, "s", int32_t, "w", float64, "d", double)
NATIVE_FTOI(tould, "u", uint32_t, "w", float64, "d", double)
NATIVE_FTOI(tosqd, "s", int64_t, "x", float64, "d", double)
NATIVE_FTOI(touqd, "u", uint64_t, "x", float64, "d", double)
NATIVE_FTOI(tosls, "s", int32_t, "w", float32, "s", float)
NATIVE_FTOI(touls, "u", uint32_t, "w", float32, "s", float)
NATIVE_FTOI(tosqs, "s", int64_t, "x", float32, "s", float)
NATIVE_FTOI(touqs, "u", uint64_t, "x", float32, "s", float)
#undef NATIVE_FTOI

#endif /* __aarch64__ */

#endif /* TARGET_ARM_NATIVE_FP_H */
