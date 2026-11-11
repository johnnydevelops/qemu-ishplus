/*
 * crypto_helper.c - emulate v8 Crypto Extensions instructions
 *
 * Copyright (C) 2013 - 2018 Linaro Ltd <ard.biesheuvel@linaro.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

#include "qemu/osdep.h"

#include "cpu.h"
#include "exec/helper-proto.h"
#include "tcg/tcg-gvec-desc.h"
#include "crypto/aes-round.h"
#include "crypto/sm4.h"
#include "vec_internal.h"

/*
 * iSH+: on an AArch64 host, the SHA instructions are the host's own (they
 * compute exactly what the guest's do); SHA-1 and SHA-256 are in every
 * AArch64 CPU with the crypto extension, SHA-512 is checked at run time.
 * AES and PMULL already use the host's (crypto/aes-round.h, clmul.h).
 */
#if defined(__aarch64__) && defined(__ARM_FEATURE_SHA2)
#include "host/cpuinfo.h"
#include <arm_neon.h>
#define NATIVE_SHA 1
#define NATIVE_V(NAME, PTR) uint32x4_t NAME; memcpy(&NAME, (PTR), 16)
#define NATIVE_DONE(VD, D, DESC) do { memcpy((VD), &(D), 16); clear_tail_16((VD), (DESC)); return; } while (0)
#endif

union CRYPTO_STATE {
    uint8_t    bytes[16];
    uint32_t   words[4];
    uint64_t   l[2];
};

#if HOST_BIG_ENDIAN
#define CR_ST_BYTE(state, i)   ((state).bytes[(15 - (i)) ^ 8])
#define CR_ST_WORD(state, i)   ((state).words[(3 - (i)) ^ 2])
#else
#define CR_ST_BYTE(state, i)   ((state).bytes[i])
#define CR_ST_WORD(state, i)   ((state).words[i])
#endif

/*
 * The caller has not been converted to full gvec, and so only
 * modifies the low 16 bytes of the vector register.
 */
static void clear_tail_16(void *vd, uint32_t desc)
{
    int opr_sz = simd_oprsz(desc);
    int max_sz = simd_maxsz(desc);

    assert(opr_sz == 16);
    clear_tail(vd, opr_sz, max_sz);
}

static const AESState aes_zero = { };

void HELPER(crypto_aese)(void *vd, void *vn, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        AESState *ad = (AESState *)(vd + i);
        AESState *st = (AESState *)(vn + i);
        AESState *rk = (AESState *)(vm + i);
        AESState t;

        /*
         * Our uint64_t are in the wrong order for big-endian.
         * The Arm AddRoundKey comes first, while the API AddRoundKey
         * comes last: perform the xor here, and provide zero to API.
         */
        if (HOST_BIG_ENDIAN) {
            t.d[0] = st->d[1] ^ rk->d[1];
            t.d[1] = st->d[0] ^ rk->d[0];
            aesenc_SB_SR_AK(&t, &t, &aes_zero, false);
            ad->d[0] = t.d[1];
            ad->d[1] = t.d[0];
        } else {
            t.v = st->v ^ rk->v;
            aesenc_SB_SR_AK(ad, &t, &aes_zero, false);
        }
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

void HELPER(crypto_aesd)(void *vd, void *vn, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        AESState *ad = (AESState *)(vd + i);
        AESState *st = (AESState *)(vn + i);
        AESState *rk = (AESState *)(vm + i);
        AESState t;

        /* Our uint64_t are in the wrong order for big-endian. */
        if (HOST_BIG_ENDIAN) {
            t.d[0] = st->d[1] ^ rk->d[1];
            t.d[1] = st->d[0] ^ rk->d[0];
            aesdec_ISB_ISR_AK(&t, &t, &aes_zero, false);
            ad->d[0] = t.d[1];
            ad->d[1] = t.d[0];
        } else {
            t.v = st->v ^ rk->v;
            aesdec_ISB_ISR_AK(ad, &t, &aes_zero, false);
        }
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

void HELPER(crypto_aesmc)(void *vd, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        AESState *ad = (AESState *)(vd + i);
        AESState *st = (AESState *)(vm + i);
        AESState t;

        /* Our uint64_t are in the wrong order for big-endian. */
        if (HOST_BIG_ENDIAN) {
            t.d[0] = st->d[1];
            t.d[1] = st->d[0];
            aesenc_MC(&t, &t, false);
            ad->d[0] = t.d[1];
            ad->d[1] = t.d[0];
        } else {
            aesenc_MC(ad, st, false);
        }
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

void HELPER(crypto_aesimc)(void *vd, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        AESState *ad = (AESState *)(vd + i);
        AESState *st = (AESState *)(vm + i);
        AESState t;

        /* Our uint64_t are in the wrong order for big-endian. */
        if (HOST_BIG_ENDIAN) {
            t.d[0] = st->d[1];
            t.d[1] = st->d[0];
            aesdec_IMC(&t, &t, false);
            ad->d[0] = t.d[1];
            ad->d[1] = t.d[0];
        } else {
            aesdec_IMC(ad, st, false);
        }
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

/*
 * SHA-1 logical functions
 */

static uint32_t cho(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & (y ^ z)) ^ z;
}

static uint32_t par(uint32_t x, uint32_t y, uint32_t z)
{
    return x ^ y ^ z;
}

static uint32_t maj(uint32_t x, uint32_t y, uint32_t z)
{
    return (x & y) | ((x | y) & z);
}

void HELPER(crypto_sha1su0)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha1su0 %0.4s, %1.4s, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *d = vd, *n = vn, *m = vm;
    uint64_t d0, d1;

    d0 = d[1] ^ d[0] ^ m[0];
    d1 = n[0] ^ d[1] ^ m[1];
    d[0] = d0;
    d[1] = d1;

    clear_tail_16(vd, desc);
}

static inline void crypto_sha1_3reg(uint64_t *rd, uint64_t *rn,
                                    uint64_t *rm, uint32_t desc,
                                    uint32_t (*fn)(union CRYPTO_STATE *d))
{
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    int i;

    for (i = 0; i < 4; i++) {
        uint32_t t = fn(&d);

        t += rol32(CR_ST_WORD(d, 0), 5) + CR_ST_WORD(n, 0)
             + CR_ST_WORD(m, i);

        CR_ST_WORD(n, 0) = CR_ST_WORD(d, 3);
        CR_ST_WORD(d, 3) = CR_ST_WORD(d, 2);
        CR_ST_WORD(d, 2) = ror32(CR_ST_WORD(d, 1), 2);
        CR_ST_WORD(d, 1) = CR_ST_WORD(d, 0);
        CR_ST_WORD(d, 0) = t;
    }
    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(rd, desc);
}

static uint32_t do_sha1c(union CRYPTO_STATE *d)
{
    return cho(CR_ST_WORD(*d, 1), CR_ST_WORD(*d, 2), CR_ST_WORD(*d, 3));
}

void HELPER(crypto_sha1c)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha1c %q0, %s1, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    crypto_sha1_3reg(vd, vn, vm, desc, do_sha1c);
}

static uint32_t do_sha1p(union CRYPTO_STATE *d)
{
    return par(CR_ST_WORD(*d, 1), CR_ST_WORD(*d, 2), CR_ST_WORD(*d, 3));
}

void HELPER(crypto_sha1p)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha1p %q0, %s1, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    crypto_sha1_3reg(vd, vn, vm, desc, do_sha1p);
}

static uint32_t do_sha1m(union CRYPTO_STATE *d)
{
    return maj(CR_ST_WORD(*d, 1), CR_ST_WORD(*d, 2), CR_ST_WORD(*d, 3));
}

void HELPER(crypto_sha1m)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha1m %q0, %s1, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    crypto_sha1_3reg(vd, vn, vm, desc, do_sha1m);
}

void HELPER(crypto_sha1h)(void *vd, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { uint32x4_t d; NATIVE_V(m, vm);
      asm("sha1h %s0, %s1" : "=w"(d) : "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rm = vm;
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };

    CR_ST_WORD(m, 0) = ror32(CR_ST_WORD(m, 0), 2);
    CR_ST_WORD(m, 1) = CR_ST_WORD(m, 2) = CR_ST_WORD(m, 3) = 0;

    rd[0] = m.l[0];
    rd[1] = m.l[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha1su1)(void *vd, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(m, vm);
      asm("sha1su1 %0.4s, %1.4s" : "+w"(d) : "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };

    CR_ST_WORD(d, 0) = rol32(CR_ST_WORD(d, 0) ^ CR_ST_WORD(m, 1), 1);
    CR_ST_WORD(d, 1) = rol32(CR_ST_WORD(d, 1) ^ CR_ST_WORD(m, 2), 1);
    CR_ST_WORD(d, 2) = rol32(CR_ST_WORD(d, 2) ^ CR_ST_WORD(m, 3), 1);
    CR_ST_WORD(d, 3) = rol32(CR_ST_WORD(d, 3) ^ CR_ST_WORD(d, 0), 1);

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

/*
 * The SHA-256 logical functions, according to
 * http://csrc.nist.gov/groups/STM/cavp/documents/shs/sha256-384-512.pdf
 */

static uint32_t S0(uint32_t x)
{
    return ror32(x, 2) ^ ror32(x, 13) ^ ror32(x, 22);
}

static uint32_t S1(uint32_t x)
{
    return ror32(x, 6) ^ ror32(x, 11) ^ ror32(x, 25);
}

static uint32_t s0(uint32_t x)
{
    return ror32(x, 7) ^ ror32(x, 18) ^ (x >> 3);
}

static uint32_t s1(uint32_t x)
{
    return ror32(x, 17) ^ ror32(x, 19) ^ (x >> 10);
}

void HELPER(crypto_sha256h)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha256h %q0, %q1, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    int i;

    for (i = 0; i < 4; i++) {
        uint32_t t = cho(CR_ST_WORD(n, 0), CR_ST_WORD(n, 1), CR_ST_WORD(n, 2))
                     + CR_ST_WORD(n, 3) + S1(CR_ST_WORD(n, 0))
                     + CR_ST_WORD(m, i);

        CR_ST_WORD(n, 3) = CR_ST_WORD(n, 2);
        CR_ST_WORD(n, 2) = CR_ST_WORD(n, 1);
        CR_ST_WORD(n, 1) = CR_ST_WORD(n, 0);
        CR_ST_WORD(n, 0) = CR_ST_WORD(d, 3) + t;

        t += maj(CR_ST_WORD(d, 0), CR_ST_WORD(d, 1), CR_ST_WORD(d, 2))
             + S0(CR_ST_WORD(d, 0));

        CR_ST_WORD(d, 3) = CR_ST_WORD(d, 2);
        CR_ST_WORD(d, 2) = CR_ST_WORD(d, 1);
        CR_ST_WORD(d, 1) = CR_ST_WORD(d, 0);
        CR_ST_WORD(d, 0) = t;
    }

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha256h2)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha256h2 %q0, %q1, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    int i;

    for (i = 0; i < 4; i++) {
        uint32_t t = cho(CR_ST_WORD(d, 0), CR_ST_WORD(d, 1), CR_ST_WORD(d, 2))
                     + CR_ST_WORD(d, 3) + S1(CR_ST_WORD(d, 0))
                     + CR_ST_WORD(m, i);

        CR_ST_WORD(d, 3) = CR_ST_WORD(d, 2);
        CR_ST_WORD(d, 2) = CR_ST_WORD(d, 1);
        CR_ST_WORD(d, 1) = CR_ST_WORD(d, 0);
        CR_ST_WORD(d, 0) = CR_ST_WORD(n, 3 - i) + t;
    }

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha256su0)(void *vd, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(m, vm);
      asm("sha256su0 %0.4s, %1.4s" : "+w"(d) : "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };

    CR_ST_WORD(d, 0) += s0(CR_ST_WORD(d, 1));
    CR_ST_WORD(d, 1) += s0(CR_ST_WORD(d, 2));
    CR_ST_WORD(d, 2) += s0(CR_ST_WORD(d, 3));
    CR_ST_WORD(d, 3) += s0(CR_ST_WORD(m, 0));

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha256su1)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    { NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
      asm("sha256su1 %0.4s, %1.4s, %2.4s" : "+w"(d) : "w"(n), "w"(m));
      NATIVE_DONE(vd, d, desc); }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };

    CR_ST_WORD(d, 0) += s1(CR_ST_WORD(m, 2)) + CR_ST_WORD(n, 1);
    CR_ST_WORD(d, 1) += s1(CR_ST_WORD(m, 3)) + CR_ST_WORD(n, 2);
    CR_ST_WORD(d, 2) += s1(CR_ST_WORD(d, 0)) + CR_ST_WORD(n, 3);
    CR_ST_WORD(d, 3) += s1(CR_ST_WORD(d, 1)) + CR_ST_WORD(m, 0);

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

/*
 * The SHA-512 logical functions (same as above but using 64-bit operands)
 */

static uint64_t cho512(uint64_t x, uint64_t y, uint64_t z)
{
    return (x & (y ^ z)) ^ z;
}

static uint64_t maj512(uint64_t x, uint64_t y, uint64_t z)
{
    return (x & y) | ((x | y) & z);
}

static uint64_t S0_512(uint64_t x)
{
    return ror64(x, 28) ^ ror64(x, 34) ^ ror64(x, 39);
}

static uint64_t S1_512(uint64_t x)
{
    return ror64(x, 14) ^ ror64(x, 18) ^ ror64(x, 41);
}

static uint64_t s0_512(uint64_t x)
{
    return ror64(x, 1) ^ ror64(x, 8) ^ (x >> 7);
}

static uint64_t s1_512(uint64_t x)
{
    return ror64(x, 19) ^ ror64(x, 61) ^ (x >> 6);
}

void HELPER(crypto_sha512h)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    if (cpuinfo & CPUINFO_SHA512) {
        NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
        asm(".arch_extension sha3\n sha512h %q0, %q1, %2.2d" : "+w"(d) : "w"(n), "w"(m));
        NATIVE_DONE(vd, d, desc);
    }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    uint64_t d0 = rd[0];
    uint64_t d1 = rd[1];

    d1 += S1_512(rm[1]) + cho512(rm[1], rn[0], rn[1]);
    d0 += S1_512(d1 + rm[0]) + cho512(d1 + rm[0], rm[1], rn[0]);

    rd[0] = d0;
    rd[1] = d1;

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha512h2)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    if (cpuinfo & CPUINFO_SHA512) {
        NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
        asm(".arch_extension sha3\n sha512h2 %q0, %q1, %2.2d" : "+w"(d) : "w"(n), "w"(m));
        NATIVE_DONE(vd, d, desc);
    }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    uint64_t d0 = rd[0];
    uint64_t d1 = rd[1];

    d1 += S0_512(rm[0]) + maj512(rn[0], rm[1], rm[0]);
    d0 += S0_512(d1) + maj512(d1, rm[0], rm[1]);

    rd[0] = d0;
    rd[1] = d1;

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha512su0)(void *vd, void *vn, uint32_t desc)
{
#ifdef NATIVE_SHA
    if (cpuinfo & CPUINFO_SHA512) {
        NATIVE_V(d, vd); NATIVE_V(n, vn);
        asm(".arch_extension sha3\n sha512su0 %0.2d, %1.2d" : "+w"(d) : "w"(n));
        NATIVE_DONE(vd, d, desc);
    }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t d0 = rd[0];
    uint64_t d1 = rd[1];

    d0 += s0_512(rd[1]);
    d1 += s0_512(rn[0]);

    rd[0] = d0;
    rd[1] = d1;

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sha512su1)(void *vd, void *vn, void *vm, uint32_t desc)
{
#ifdef NATIVE_SHA
    if (cpuinfo & CPUINFO_SHA512) {
        NATIVE_V(d, vd); NATIVE_V(n, vn); NATIVE_V(m, vm);
        asm(".arch_extension sha3\n sha512su1 %0.2d, %1.2d, %2.2d" : "+w"(d) : "w"(n), "w"(m));
        NATIVE_DONE(vd, d, desc);
    }
#endif

    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;

    rd[0] += s1_512(rn[0]) + rm[0];
    rd[1] += s1_512(rn[1]) + rm[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sm3partw1)(void *vd, void *vn, void *vm, uint32_t desc)
{
    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    uint32_t t;

    t = CR_ST_WORD(d, 0) ^ CR_ST_WORD(n, 0) ^ ror32(CR_ST_WORD(m, 1), 17);
    CR_ST_WORD(d, 0) = t ^ ror32(t, 17) ^ ror32(t, 9);

    t = CR_ST_WORD(d, 1) ^ CR_ST_WORD(n, 1) ^ ror32(CR_ST_WORD(m, 2), 17);
    CR_ST_WORD(d, 1) = t ^ ror32(t, 17) ^ ror32(t, 9);

    t = CR_ST_WORD(d, 2) ^ CR_ST_WORD(n, 2) ^ ror32(CR_ST_WORD(m, 3), 17);
    CR_ST_WORD(d, 2) = t ^ ror32(t, 17) ^ ror32(t, 9);

    t = CR_ST_WORD(d, 3) ^ CR_ST_WORD(n, 3) ^ ror32(CR_ST_WORD(d, 0), 17);
    CR_ST_WORD(d, 3) = t ^ ror32(t, 17) ^ ror32(t, 9);

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

void HELPER(crypto_sm3partw2)(void *vd, void *vn, void *vm, uint32_t desc)
{
    uint64_t *rd = vd;
    uint64_t *rn = vn;
    uint64_t *rm = vm;
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    uint32_t t = CR_ST_WORD(n, 0) ^ ror32(CR_ST_WORD(m, 0), 25);

    CR_ST_WORD(d, 0) ^= t;
    CR_ST_WORD(d, 1) ^= CR_ST_WORD(n, 1) ^ ror32(CR_ST_WORD(m, 1), 25);
    CR_ST_WORD(d, 2) ^= CR_ST_WORD(n, 2) ^ ror32(CR_ST_WORD(m, 2), 25);
    CR_ST_WORD(d, 3) ^= CR_ST_WORD(n, 3) ^ ror32(CR_ST_WORD(m, 3), 25) ^
                        ror32(t, 17) ^ ror32(t, 2) ^ ror32(t, 26);

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(vd, desc);
}

static inline void QEMU_ALWAYS_INLINE
crypto_sm3tt(uint64_t *rd, uint64_t *rn, uint64_t *rm,
             uint32_t desc, uint32_t opcode)
{
    union CRYPTO_STATE d = { .l = { rd[0], rd[1] } };
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    uint32_t imm2 = simd_data(desc);
    uint32_t t;

    assert(imm2 < 4);

    if (opcode == 0 || opcode == 2) {
        /* SM3TT1A, SM3TT2A */
        t = par(CR_ST_WORD(d, 3), CR_ST_WORD(d, 2), CR_ST_WORD(d, 1));
    } else if (opcode == 1) {
        /* SM3TT1B */
        t = maj(CR_ST_WORD(d, 3), CR_ST_WORD(d, 2), CR_ST_WORD(d, 1));
    } else if (opcode == 3) {
        /* SM3TT2B */
        t = cho(CR_ST_WORD(d, 3), CR_ST_WORD(d, 2), CR_ST_WORD(d, 1));
    } else {
        qemu_build_not_reached();
    }

    t += CR_ST_WORD(d, 0) + CR_ST_WORD(m, imm2);

    CR_ST_WORD(d, 0) = CR_ST_WORD(d, 1);

    if (opcode < 2) {
        /* SM3TT1A, SM3TT1B */
        t += CR_ST_WORD(n, 3) ^ ror32(CR_ST_WORD(d, 3), 20);

        CR_ST_WORD(d, 1) = ror32(CR_ST_WORD(d, 2), 23);
    } else {
        /* SM3TT2A, SM3TT2B */
        t += CR_ST_WORD(n, 3);
        t ^= rol32(t, 9) ^ rol32(t, 17);

        CR_ST_WORD(d, 1) = ror32(CR_ST_WORD(d, 2), 13);
    }

    CR_ST_WORD(d, 2) = CR_ST_WORD(d, 3);
    CR_ST_WORD(d, 3) = t;

    rd[0] = d.l[0];
    rd[1] = d.l[1];

    clear_tail_16(rd, desc);
}

#define DO_SM3TT(NAME, OPCODE) \
    void HELPER(NAME)(void *vd, void *vn, void *vm, uint32_t desc) \
    { crypto_sm3tt(vd, vn, vm, desc, OPCODE); }

DO_SM3TT(crypto_sm3tt1a, 0)
DO_SM3TT(crypto_sm3tt1b, 1)
DO_SM3TT(crypto_sm3tt2a, 2)
DO_SM3TT(crypto_sm3tt2b, 3)

#undef DO_SM3TT

static void do_crypto_sm4e(uint64_t *rd, uint64_t *rn, uint64_t *rm)
{
    union CRYPTO_STATE d = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE n = { .l = { rm[0], rm[1] } };
    uint32_t t, i;

    for (i = 0; i < 4; i++) {
        t = CR_ST_WORD(d, (i + 1) % 4) ^
            CR_ST_WORD(d, (i + 2) % 4) ^
            CR_ST_WORD(d, (i + 3) % 4) ^
            CR_ST_WORD(n, i);

        t = sm4_subword(t);

        CR_ST_WORD(d, i) ^= t ^ rol32(t, 2) ^ rol32(t, 10) ^ rol32(t, 18) ^
                            rol32(t, 24);
    }

    rd[0] = d.l[0];
    rd[1] = d.l[1];
}

void HELPER(crypto_sm4e)(void *vd, void *vn, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        do_crypto_sm4e(vd + i, vn + i, vm + i);
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

static void do_crypto_sm4ekey(uint64_t *rd, uint64_t *rn, uint64_t *rm)
{
    union CRYPTO_STATE d;
    union CRYPTO_STATE n = { .l = { rn[0], rn[1] } };
    union CRYPTO_STATE m = { .l = { rm[0], rm[1] } };
    uint32_t t, i;

    d = n;
    for (i = 0; i < 4; i++) {
        t = CR_ST_WORD(d, (i + 1) % 4) ^
            CR_ST_WORD(d, (i + 2) % 4) ^
            CR_ST_WORD(d, (i + 3) % 4) ^
            CR_ST_WORD(m, i);

        t = sm4_subword(t);

        CR_ST_WORD(d, i) ^= t ^ rol32(t, 13) ^ rol32(t, 23);
    }

    rd[0] = d.l[0];
    rd[1] = d.l[1];
}

void HELPER(crypto_sm4ekey)(void *vd, void *vn, void* vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);

    for (i = 0; i < opr_sz; i += 16) {
        do_crypto_sm4ekey(vd + i, vn + i, vm + i);
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}

void HELPER(crypto_rax1)(void *vd, void *vn, void *vm, uint32_t desc)
{
    intptr_t i, opr_sz = simd_oprsz(desc);
    uint64_t *d = vd, *n = vn, *m = vm;

    for (i = 0; i < opr_sz / 8; ++i) {
        d[i] = n[i] ^ rol64(m[i], 1);
    }
    clear_tail(vd, opr_sz, simd_maxsz(desc));
}
