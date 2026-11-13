/*
 * ishjit: operand decoder for the A64 "Data Processing -- Scalar
 * Floating-Point and Advanced SIMD" encoding space.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef ISHJIT_SIMD_H
#define ISHJIT_SIMD_H

#include <stdint.h>
#include <stdbool.h>

/* Values of SimdOperands.rd/rn/rm */
#define SIMD_NOT_GPR    0   /* field is a V register, an immediate or opcode */
#define SIMD_GPR_ZR     1   /* GPR, 31 is XZR/WZR */
#define SIMD_GPR_SP     2   /* GPR, 31 is SP (never produced in this space) */

typedef struct {
    uint8_t rd, rn, rm;     /* fields [4:0], [9:5], [20:16] */
    bool rd_written;        /* rd is a GPR and is written (never read) */
    bool reads_nzcv, writes_nzcv;
    bool reads_fpcr_mode;   /* result depends on FPCR (RMode, FZ, FZ16, DN, AHP) */
    bool sets_qc;           /* may set FPSR.QC (saturating integer SIMD) */
} SimdOperands;

bool ishjit_simd_decode(uint32_t insn, SimdOperands *g);

#endif
