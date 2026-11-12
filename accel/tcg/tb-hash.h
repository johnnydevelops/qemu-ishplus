/*
 * internal execution defines for qemu
 *
 *  Copyright (c) 2003 Fabrice Bellard
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

#ifndef EXEC_TB_HASH_H
#define EXEC_TB_HASH_H

#include "exec/cpu-defs.h"
#include "exec/exec-all.h"
#include "exec/translation-block.h"
#include "qemu/xxhash.h"
#include "tb-jmp-cache.h"

#ifdef CONFIG_SOFTMMU

/*
 * In softmmu, the jump cache is split in groups, one per page (and other
 * pages hashing to it): a TLB page flush clears the page's group.
 * iSH+: 256 groups of 256 entries. With 64 entries a group, the TBs of
 * one page evicted each other: most misses were such collisions. The
 * entry within the group mixes the insn's offset with higher bits.
 */
#define TB_JMP_PAGE_BITS 8
#define TB_JMP_PAGE_SIZE (1 << TB_JMP_PAGE_BITS)
#define TB_JMP_ADDR_MASK (TB_JMP_PAGE_SIZE - 1)
#define TB_JMP_GROUP_BITS (TB_JMP_CACHE_BITS - TB_JMP_PAGE_BITS)
#define TB_JMP_PAGE_MASK (TB_JMP_CACHE_SIZE - TB_JMP_PAGE_SIZE)

/* The group of the page of @pc (from its page number only) */
static inline unsigned int tb_jmp_cache_hash_page(vaddr pc)
{
    vaddr page = pc >> TARGET_PAGE_BITS;

    page ^= (page >> TB_JMP_GROUP_BITS) ^ (page >> (2 * TB_JMP_GROUP_BITS));
    return (page << TB_JMP_PAGE_BITS) & TB_JMP_PAGE_MASK;
}

static inline unsigned int tb_jmp_cache_hash_func(vaddr pc)
{
    vaddr e = (pc >> 2) ^ (pc >> (2 + TB_JMP_PAGE_BITS)) ^
              (pc >> (TARGET_PAGE_BITS + TB_JMP_GROUP_BITS));

    return tb_jmp_cache_hash_page(pc) | (e & TB_JMP_ADDR_MASK);
}

#else

/* In user-mode we can get better hashing because we do not have a TLB */
static inline unsigned int tb_jmp_cache_hash_func(vaddr pc)
{
    return (pc ^ (pc >> TB_JMP_CACHE_BITS)) & (TB_JMP_CACHE_SIZE - 1);
}

#endif /* CONFIG_SOFTMMU */

static inline
uint32_t tb_hash_func(tb_page_addr_t phys_pc, vaddr pc,
                      uint32_t flags, uint64_t flags2, uint32_t cf_mask)
{
    return qemu_xxhash8(phys_pc, pc, flags2, flags, cf_mask);
}

#endif
