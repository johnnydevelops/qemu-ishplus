/*
 * Copyright (c) 2019 Linaro
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or
 * (at your option) any later version.
 *
 * See the COPYING file in the top-level directory for details.
 *
 * Target-specific opcodes for host vector expansion.  These will be
 * emitted by tcg_expand_vec_op.  For those familiar with GCC internals,
 * consider these to be UNSPEC with names.
 */

DEF(aa64_sshl_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_sli_vec, 1, 2, 1, TCG_OPF_VECTOR)
/*
 * iSH+: floating point for AArch64 guests (target/arm/tcg/translate-a64.c),
 * emitted only when the guest's FPCR is the host's (all default): the
 * result and exception flags are the guest's. Type V128 is a vector
 * (MO_32: 4 singles, MO_64: 2 doubles); V64 is a scalar (MO_32: single,
 * MO_64: double). fmla: d = a + n * m fused, d and a the same register.
 */
DEF(aa64_fadd_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fsub_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fmul_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fdiv_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fmax_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fmin_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fmaxnm_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fminnm_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fabd_vec, 1, 2, 0, TCG_OPF_VECTOR)
DEF(aa64_fnmul_vec, 1, 2, 0, TCG_OPF_VECTOR)
/* iSH+: a 128-bit guest load into a vector register (like qemu_ld_i128) */
DEF(aa64_qemu_ld_v128, 1, 1, 1,
    TCG_OPF_VECTOR | TCG_OPF_CALL_CLOBBER | TCG_OPF_SIDE_EFFECTS)
DEF(aa64_qemu_st_v128, 0, 2, 1,
    TCG_OPF_VECTOR | TCG_OPF_CALL_CLOBBER | TCG_OPF_SIDE_EFFECTS)
DEF(aa64_fsqrt_vec, 1, 1, 0, TCG_OPF_VECTOR)
DEF(aa64_fmla_vec, 1, 3, 0, TCG_OPF_VECTOR)
DEF(aa64_fmls_vec, 1, 3, 0, TCG_OPF_VECTOR)
DEF(aa64_fnmla_vec, 1, 3, 0, TCG_OPF_VECTOR)
DEF(aa64_fnmls_vec, 1, 3, 0, TCG_OPF_VECTOR)
