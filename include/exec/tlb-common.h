/*
 * Common definitions for the softmmu tlb
 *
 * Copyright (c) 2003 Fabrice Bellard
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, see <http://www.gnu.org/licenses/>.
 */
#ifndef EXEC_TLB_COMMON_H
#define EXEC_TLB_COMMON_H 1

#define CPU_TLB_ENTRY_BITS (HOST_LONG_BITS == 32 ? 4 : 5)

/* Minimalized TLB entry for use by TCG fast path. */
typedef union CPUTLBEntry {
    /*
     * iSH+: the addend right after addr_read and right before addr_write,
     * so that a load's or a store's check reads its comparator and the
     * addend with one ldp (TLB_ENTRY_IDX maps an access type to its slot).
     */
    struct {
        uintptr_t addr_read;
        /*
         * Addend to virtual address to get host address.  IO accesses
         * use the corresponding iotlb value.
         */
        uintptr_t addend;
        uintptr_t addr_write;
        uintptr_t addr_code;
    };
    /*
     * Padding to get a power of two size, as well as index
     * access to addr_{read,write,code}.
     */
    uintptr_t addr_idx[(1 << CPU_TLB_ENTRY_BITS) / sizeof(uintptr_t)];
} CPUTLBEntry;

QEMU_BUILD_BUG_ON(sizeof(CPUTLBEntry) != (1 << CPU_TLB_ENTRY_BITS));

/* the addr_idx[] slot of an access type's comparator (load 0, store 2, code 3) */
#define TLB_ENTRY_IDX(access_type) ((access_type) + ((access_type) != 0))

/*
 * Data elements that are per MMU mode, accessed by the fast path.
 * The structure is aligned to aid loading the pair with one insn.
 */
typedef struct CPUTLBDescFast {
    /* Contains (n_entries - 1) << CPU_TLB_ENTRY_BITS */
    uintptr_t mask;
    /* The array of tlb entries itself. */
    CPUTLBEntry *table;
} CPUTLBDescFast QEMU_ALIGNED(2 * sizeof(void *));

#endif /* EXEC_TLB_COMMON_H */
