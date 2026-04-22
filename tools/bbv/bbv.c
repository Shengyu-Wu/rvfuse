/*
 * Generate basic block vectors for use with the SimPoint analysis tool.
 * SimPoint: https://cseweb.ucsd.edu/~calder/simpoint/
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Extended: .disas file output, plugin_flush() at exit,
 * and function-scoped recording via symbol/addr/syscall detection.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glib.h>

#include <qemu-plugin.h>

typedef struct Bb {
    uint64_t vaddr;
    struct qemu_plugin_scoreboard *count;
    unsigned int index;
} Bb;

typedef struct Vcpu {
    uint64_t count;
    FILE *file;
} Vcpu;

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;
static GHashTable *bbs;
static GRWLock bbs_lock;
static char *filename;
static FILE *disas_file;
static struct qemu_plugin_scoreboard *vcpus;
static uint64_t interval = 100000000;

/* ========== Function-Scoped Recording ========== */

enum plugin_state {
    STATE_DETECTING,    /* Waiting for target symbol / library mmap */
    STATE_RECORDING     /* Recording only target function BBs */
};

/* Configuration (user-provided via plugin args) */
static char *target_func_name;      /* e.g. "ggml_gemv_q4_0_16x1_q8_0" */
static char *lib_name;              /* Shared library name, e.g. "libggml-cpu" */
static uint64_t func_offset;        /* Static offset from nm, e.g. 0xac9a4 */
static uint64_t target_func_size;   /* e.g. 0x30a (778 bytes) */
static uint64_t func_addr;          /* Direct runtime address */

/* Detected/calculated at runtime */
static uint64_t func_start_vaddr;
static uint64_t func_end_vaddr;

/* State tracking */
static enum plugin_state state;
static bool filter_enabled;
static uint64_t detect_insn_count;
static int symbol_match_count;

#define MAX_DETECT_INSNS 100000

/* Header tracking for disas output */
static bool disas_header_written;

/* ========== Syscall-Based Library Detection ========== */

static GHashTable *tracked_fds;
static GByteArray *read_buf;

/* Pending syscall state (correlate entry <-> return) */
static int64_t pending_syscall_num;
static bool pending_openat_is_target;
static uint64_t pending_mmap_fd;
static uint64_t pending_mmap_prot;
static uint64_t pending_mmap_offset;

#define SYS_OPENAT_RV64  56
#define SYS_MMAP_RV64    222
#define PROT_EXEC_FLAG   4

static void free_bb(void *data)
{
    qemu_plugin_scoreboard_free(((Bb *)data)->count);
    g_free(data);
}

/*
 * Syscall entry callback.
 * Tracks openat (56) to find when the target library is opened,
 * and mmap (222) to save arguments for use in the return callback.
 *
 * RISC-V syscall arg mapping:
 *   callback a1..a8 = guest a0..a7
 *   openat: a2(guest a1)=pathname, mmap: a3(guest a2)=prot, a5(guest a4)=fd, a6(guest a5)=offset
 */
static void vcpu_syscall(qemu_plugin_id_t id, unsigned int vcpu_index,
                         int64_t num, uint64_t a1, uint64_t a2,
                         uint64_t a3, uint64_t a4, uint64_t a5,
                         uint64_t a6, uint64_t a7, uint64_t a8)
{
    if (state == STATE_RECORDING) return;

    pending_syscall_num = num;

    if (num == SYS_OPENAT_RV64 && lib_name) {
        g_byte_array_set_size(read_buf, 0);
        if (qemu_plugin_read_memory_vaddr(a2, read_buf, 256)) {
            g_byte_array_append(read_buf, (const guint8 *)"", 1);
            pending_openat_is_target = strstr((const char *)read_buf->data, lib_name) != NULL;
        } else {
            pending_openat_is_target = false;
        }
    }

    if (num == SYS_MMAP_RV64) {
        pending_mmap_fd = a5;
        pending_mmap_prot = a3;
        pending_mmap_offset = a6;
    }
}

/*
 * Syscall return callback.
 * Completes openat tracking (fd -> library) and mmap detection (base address).
 *
 * Strategy:
 *   1. First mmap (offset=0) for tracked fd maps LOAD[0] at the ELF base.
 *      For PIC shared libs, LOAD[0] has p_vaddr=0, so ret IS the base.
 *   2. Wait for the RE (PROT_EXEC) mmap to confirm the text segment is mapped.
 *   3. func_addr = elf_base + nm_offset (nm offsets are virtual addresses).
 */
static void vcpu_syscall_ret(qemu_plugin_id_t id, unsigned int vcpu_idx,
                             int64_t num, int64_t ret)
{
    if (state == STATE_RECORDING) return;

    /* openat return: record fd if it belongs to our target library */
    if (num == SYS_OPENAT_RV64 && pending_openat_is_target && ret >= 0) {
        g_hash_table_add(tracked_fds, GINT_TO_POINTER((int)ret));
        pending_openat_is_target = false;
    }

    static uint64_t lib_base_addr = 0;
    static bool base_detected = false;

    if (num == SYS_MMAP_RV64 && ret > 0 &&
        g_hash_table_contains(tracked_fds, GINT_TO_POINTER((int)pending_mmap_fd))) {

        if (!base_detected && pending_mmap_offset == 0) {
            lib_base_addr = (uint64_t)ret;
            base_detected = true;
        }

        if ((pending_mmap_prot & PROT_EXEC_FLAG) && base_detected) {
            /* Text segment mapped. Calculate function address from ELF base. */
            func_start_vaddr = lib_base_addr + func_offset;
            func_end_vaddr = func_start_vaddr + target_func_size;
            state = STATE_RECORDING;

            if (disas_file && !disas_header_written) {
                uint64_t text_seg_p_vaddr = (uint64_t)ret - lib_base_addr;
                fprintf(disas_file, "# BBV Function-Scoped Mode (Syscall-Based)\n");
                fprintf(disas_file, "# Library: %s (base 0x%" PRIx64 ")\n",
                        lib_name, lib_base_addr);
                fprintf(disas_file, "# Text segment: p_vaddr 0x%" PRIx64
                        ", file_offset 0x%" PRIx64 "\n",
                        text_seg_p_vaddr, pending_mmap_offset);
                fprintf(disas_file, "# Function: offset 0x%" PRIx64
                        ", size 0x%" PRIx64 "\n",
                        func_offset, target_func_size);
                fprintf(disas_file, "# Range: 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                        func_start_vaddr, func_end_vaddr);
                fprintf(disas_file, "#\n\n");
                fflush(disas_file);
                disas_header_written = true;
            }

            g_autofree gchar *msg = g_strdup_printf(
                "BBV: Library '%s' mapped at 0x%" PRIx64
                ", function at 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                lib_name, lib_base_addr, func_start_vaddr, func_end_vaddr);
            qemu_plugin_outs(msg);
        }
    }
}

static qemu_plugin_u64 count_u64(void)
{
    return qemu_plugin_scoreboard_u64_in_struct(vcpus, Vcpu, count);
}

static qemu_plugin_u64 bb_count_u64(Bb *bb)
{
    return qemu_plugin_scoreboard_u64(bb->count);
}

static void plugin_flush(void)
{
    for (int i = 0; i < qemu_plugin_num_vcpus(); i++) {
        Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, i);
        GHashTableIter iter;
        void *value;

        if (!vcpu->file || !vcpu->count) {
            continue;
        }

        fputc('T', vcpu->file);

        g_rw_lock_reader_lock(&bbs_lock);
        g_hash_table_iter_init(&iter, bbs);

        while (g_hash_table_iter_next(&iter, NULL, &value)) {
            Bb *bb = value;
            uint64_t bb_count = qemu_plugin_u64_get(bb_count_u64(bb), i);

            if (!bb_count) {
                continue;
            }

            fprintf(vcpu->file, ":%u:%" PRIu64 " ", bb->index, bb_count);
            qemu_plugin_u64_set(bb_count_u64(bb), i, 0);
        }

        g_rw_lock_reader_unlock(&bbs_lock);
        fputc('\n', vcpu->file);
    }
}

static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    if (filter_enabled && state == STATE_DETECTING) {
        const char *target_name = target_func_name ? target_func_name :
                                  (lib_name ? lib_name : "(unknown)");
        g_autofree gchar *msg = g_strdup_printf(
            "BBV WARNING: Target function '%s' never executed. No BBV data recorded.\n",
            target_name);
        qemu_plugin_outs(msg);

        if (disas_file) {
            fprintf(disas_file, "# BBV Function-Scoped Mode - TARGET NOT FOUND\n");
            fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                    target_name, target_func_size);
            if (lib_name) {
                fprintf(disas_file, "# Library: %s, Offset: 0x%" PRIx64 "\n",
                        lib_name, func_offset);
            }
            fprintf(disas_file, "# Symbol matches: %d\n", symbol_match_count);
            fprintf(disas_file, "# Instructions checked: %" PRIu64 "\n", detect_insn_count);
        }
    }

    plugin_flush();

    for (int i = 0; i < qemu_plugin_num_vcpus(); i++) {
        Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, i);
        if (vcpu->file) {
            fclose(vcpu->file);
        }
    }

    g_hash_table_unref(bbs);
    g_free(filename);
    g_free(target_func_name);
    g_free(lib_name);
    if (tracked_fds) {
        g_hash_table_destroy(tracked_fds);
    }
    if (read_buf) {
        g_byte_array_free(read_buf, TRUE);
    }
    if (disas_file) {
        fclose(disas_file);
    }
    qemu_plugin_scoreboard_free(vcpus);
}

static void vcpu_init(qemu_plugin_id_t id, unsigned int vcpu_index)
{
    g_autofree gchar *vcpu_filename = NULL;
    Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, vcpu_index);

    vcpu_filename = g_strdup_printf("%s.%u.bb", filename, vcpu_index);
    vcpu->file = fopen(vcpu_filename, "w");
    if (!vcpu->file) {
        fprintf(stderr, "bbv: failed to open %s for writing\n", vcpu_filename);
    }
}

static void vcpu_interval_exec(unsigned int vcpu_index, void *udata)
{
    Vcpu *vcpu = qemu_plugin_scoreboard_find(vcpus, vcpu_index);
    GHashTableIter iter;
    void *value;

    if (!vcpu->file) {
        return;
    }

    vcpu->count -= interval;

    fputc('T', vcpu->file);

    g_rw_lock_reader_lock(&bbs_lock);
    g_hash_table_iter_init(&iter, bbs);

    while (g_hash_table_iter_next(&iter, NULL, &value)) {
        Bb *bb = value;
        uint64_t bb_count = qemu_plugin_u64_get(bb_count_u64(bb), vcpu_index);

        if (!bb_count) {
            continue;
        }

        fprintf(vcpu->file, ":%u:%" PRIu64 " ", bb->index, bb_count);
        qemu_plugin_u64_set(bb_count_u64(bb), vcpu_index, 0);
    }

    g_rw_lock_reader_unlock(&bbs_lock);
    fputc('\n', vcpu->file);
}

/* ========== RISC-V Prologue Detection ========== */

static bool is_stack_alloc_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        uint16_t insn = (uint16_t)insn_raw;
        uint16_t funct3 = (insn >> 13) & 0x7;
        uint16_t rd = (insn >> 7) & 0x1F;
        uint16_t quadrant = insn & 0x3;
        return funct3 == 3 && rd == 2 && quadrant == 1;
    } else if (insn_size == 4) {
        uint32_t opcode = insn_raw & 0x7F;
        uint32_t rd = (insn_raw >> 7) & 0x1F;
        uint32_t rs1 = (insn_raw >> 15) & 0x1F;
        uint32_t funct3 = (insn_raw >> 12) & 0x7;
        int32_t imm = (int32_t)((insn_raw >> 20) & 0xFFF);
        if (imm & 0x800) imm |= 0xFFFFF000;
        return opcode == 0x13 && funct3 == 0x0 && rd == 2 && rs1 == 2 && imm < 0;
    }
    return false;
}

static bool is_callee_save_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        uint16_t insn = (uint16_t)insn_raw;
        uint16_t quadrant = insn & 0x3;
        uint16_t funct3 = (insn >> 13) & 0x7;
        return quadrant == 2 && funct3 == 7;
    } else if (insn_size == 4) {
        uint32_t opcode = insn_raw & 0x7F;
        uint32_t funct3 = (insn_raw >> 12) & 0x7;
        uint32_t rs1 = (insn_raw >> 15) & 0x1F;
        return opcode == 0x23 && funct3 == 0x3 && rs1 == 2;
    }
    return false;
}

__attribute__((unused))
static bool is_riscv_function_entry(struct qemu_plugin_tb *tb)
{
    size_t n_insns = qemu_plugin_tb_n_insns(tb);
    if (n_insns < 2) return false;

    struct qemu_plugin_insn *insn0 = qemu_plugin_tb_get_insn(tb, 0);
    size_t size0 = qemu_plugin_insn_size(insn0);
    uint8_t data0[4] = {0};
    qemu_plugin_insn_data(insn0, data0, size0);
    uint32_t raw0 = (size0 == 2) ? *(uint16_t *)data0 : *(uint32_t *)data0;

    if (!is_stack_alloc_insn(raw0, size0)) {
        struct qemu_plugin_insn *insn1 = qemu_plugin_tb_get_insn(tb, 1);
        size_t size1 = qemu_plugin_insn_size(insn1);
        uint8_t data1[4] = {0};
        qemu_plugin_insn_data(insn1, data1, size1);
        uint32_t raw1 = (size1 == 2) ? *(uint16_t *)data1 : *(uint32_t *)data1;
        return is_callee_save_insn(raw1, size1);
    }

    struct qemu_plugin_insn *insn1 = qemu_plugin_tb_get_insn(tb, 1);
    size_t size1 = qemu_plugin_insn_size(insn1);
    uint8_t data1[4] = {0};
    qemu_plugin_insn_data(insn1, data1, size1);
    uint32_t raw1 = (size1 == 2) ? *(uint16_t *)data1 : *(uint32_t *)data1;

    return is_callee_save_insn(raw1, size1) || n_insns > 2;
}

/*
 * Detect target function by:
 * 1. Symbol name matching (for statically linked or main program symbols)
 * 2. Direct address mode (func_addr + func_size specified)
 * Method 3 (lib_name + func_offset) is handled by syscall callbacks.
 */
static void detect_target_symbol(struct qemu_plugin_tb *tb)
{
    size_t n_insns = qemu_plugin_tb_n_insns(tb);
    uint64_t tb_vaddr = qemu_plugin_tb_vaddr(tb);

    detect_insn_count += n_insns;

    if (detect_insn_count > MAX_DETECT_INSNS && state == STATE_DETECTING) {
        qemu_plugin_outs("BBV: Detection timeout - checking final symbols\n");
    }

    /* Method 1: Symbol name matching */
    if (target_func_name) {
        for (size_t i = 0; i < n_insns; i++) {
            struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
            const char *sym = qemu_plugin_insn_symbol(insn);

            if (sym && g_strcmp0(sym, target_func_name) == 0) {
                symbol_match_count++;
                bool is_entry = is_riscv_function_entry(tb);

                if (is_entry || detect_insn_count > MAX_DETECT_INSNS) {
                    func_start_vaddr = tb_vaddr;
                    func_end_vaddr = func_start_vaddr + target_func_size;
                    state = STATE_RECORDING;

                    if (disas_file && !disas_header_written) {
                        fprintf(disas_file, "# BBV Function-Scoped Mode\n");
                        fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                                target_func_name, target_func_size);
                        fprintf(disas_file, "# Range: 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                                func_start_vaddr, func_end_vaddr);
                        fprintf(disas_file, "#\n\n");
                        disas_header_written = true;
                    }

                    g_autofree gchar *msg = g_strdup_printf(
                        "BBV: Target function '%s' detected at 0x%" PRIx64
                        " (size 0x%" PRIx64 ", end 0x%" PRIx64 ")%s\n",
                        target_func_name, func_start_vaddr, target_func_size,
                        func_end_vaddr, is_entry ? "" : " [timeout fallback]");
                    qemu_plugin_outs(msg);
                    return;
                }
            }
        }
    }

    /* Method 2: Direct address mode */
    if (func_addr > 0 && target_func_size > 0 && state == STATE_DETECTING) {
        func_start_vaddr = func_addr;
        func_end_vaddr = func_addr + target_func_size;
        state = STATE_RECORDING;

        if (disas_file && !disas_header_written) {
            fprintf(disas_file, "# BBV Function-Scoped Mode (Direct Address)\n");
            fprintf(disas_file, "# Function address: 0x%" PRIx64 "\n", func_addr);
            fprintf(disas_file, "# Function size: 0x%" PRIx64 "\n", target_func_size);
            fprintf(disas_file, "# Range: 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                    func_start_vaddr, func_end_vaddr);
            fprintf(disas_file, "#\n\n");
            disas_header_written = true;
        }

        g_autofree gchar *msg = g_strdup_printf(
            "BBV: Direct address mode - recording [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
            func_start_vaddr, func_end_vaddr);
        qemu_plugin_outs(msg);
        return;
    }

    /* Method 3: lib_name + func_offset — handled by syscall callbacks */
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    uint64_t n_insns = qemu_plugin_tb_n_insns(tb);
    uint64_t vaddr = qemu_plugin_tb_vaddr(tb);
    Bb *bb;

    /* ========== Filter Mode: State Machine ========== */
    if (filter_enabled) {
        switch (state) {
        case STATE_DETECTING:
            detect_target_symbol(tb);
            return;

        case STATE_RECORDING:
            if (vaddr < func_start_vaddr || vaddr >= func_end_vaddr) {
                return;
            }
            break;
        }
    }

    /* ========== Normal Instrumentation ========== */

    g_rw_lock_writer_lock(&bbs_lock);
    bb = g_hash_table_lookup(bbs, &vaddr);
    if (!bb) {
        bb = g_new(Bb, 1);
        bb->vaddr = vaddr;
        bb->count = qemu_plugin_scoreboard_new(sizeof(uint64_t));
        bb->index = g_hash_table_size(bbs);
        g_hash_table_replace(bbs, &bb->vaddr, bb);

        if (disas_file) {
            fprintf(disas_file, "BB %u (vaddr: 0x%" PRIx64 ", %" PRIu64 " insns):\n",
                    bb->index, bb->vaddr, n_insns);
            for (size_t i = 0; i < n_insns; i++) {
                struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
                uint64_t insn_vaddr = qemu_plugin_insn_vaddr(insn);
                char *disas = qemu_plugin_insn_disas(insn);
                fprintf(disas_file, "  0x%" PRIx64 ": %s\n",
                        insn_vaddr, disas ? disas : "unknown");
                g_free(disas);
            }
            fprintf(disas_file, "\n");
        }
    }
    g_rw_lock_writer_unlock(&bbs_lock);

    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, count_u64(), n_insns);

    qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
        tb, QEMU_PLUGIN_INLINE_ADD_U64, bb_count_u64(bb), n_insns);

    qemu_plugin_register_vcpu_tb_exec_cond_cb(
        tb, vcpu_interval_exec, QEMU_PLUGIN_CB_NO_REGS,
        QEMU_PLUGIN_COND_GE, count_u64(), interval, NULL);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        char *opt = argv[i];
        g_auto(GStrv) tokens = g_strsplit(opt, "=", 2);
        if (g_strcmp0(tokens[0], "interval") == 0) {
            interval = g_ascii_strtoull(tokens[1], NULL, 10);
        } else if (g_strcmp0(tokens[0], "outfile") == 0) {
            filename = tokens[1];
            tokens[1] = NULL;
        } else if (g_strcmp0(tokens[0], "func_name") == 0) {
            target_func_name = g_strdup(tokens[1]);
            filter_enabled = true;
            state = STATE_DETECTING;
            detect_insn_count = 0;
            symbol_match_count = 0;
            disas_header_written = false;
            tokens[1] = NULL;
        } else if (g_strcmp0(tokens[0], "func_size") == 0) {
            target_func_size = g_ascii_strtoull(tokens[1], NULL, 0);
        } else if (g_strcmp0(tokens[0], "lib_name") == 0) {
            lib_name = g_strdup(tokens[1]);
            filter_enabled = true;
            state = STATE_DETECTING;
            detect_insn_count = 0;
            tokens[1] = NULL;
        } else if (g_strcmp0(tokens[0], "func_offset") == 0) {
            func_offset = g_ascii_strtoull(tokens[1], NULL, 0);
        } else if (g_strcmp0(tokens[0], "func_addr") == 0) {
            func_addr = g_ascii_strtoull(tokens[1], NULL, 0);
            filter_enabled = true;
            state = STATE_DETECTING;
            detect_insn_count = 0;
        } else {
            fprintf(stderr, "option parsing failed: %s\n", opt);
            return -1;
        }
    }

    if (!filename) {
        fputs("outfile unspecified\n", stderr);
        return -1;
    }

    if (filter_enabled && !target_func_name && !lib_name && func_addr == 0) {
        fputs("func_name, lib_name, or func_addr required for filter mode\n", stderr);
        return -1;
    }

    if (func_addr > 0 && target_func_size == 0) {
        fputs("func_size required when func_addr is specified\n", stderr);
        return -1;
    }

    if (lib_name && func_offset == 0) {
        fputs("func_offset required when lib_name is specified\n", stderr);
        return -1;
    }

    g_autofree gchar *disas_filename = g_strdup_printf("%s.disas", filename);
    disas_file = fopen(disas_filename, "w");
    if (!disas_file) {
        fprintf(stderr, "bbv: failed to open %s for writing — disassembly output disabled\n",
                disas_filename);
    }

    bbs = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, free_bb);
    vcpus = qemu_plugin_scoreboard_new(sizeof(Vcpu));
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    qemu_plugin_register_vcpu_init_cb(id, vcpu_init);
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);

    /* Register syscall callbacks for library detection */
    if (lib_name && func_offset > 0) {
        tracked_fds = g_hash_table_new(g_direct_hash, g_direct_equal);
        read_buf = g_byte_array_new();
        qemu_plugin_register_vcpu_syscall_cb(id, vcpu_syscall);
        qemu_plugin_register_vcpu_syscall_ret_cb(id, vcpu_syscall_ret);
    }

    return 0;
}
