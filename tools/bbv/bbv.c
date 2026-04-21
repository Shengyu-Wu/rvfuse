/*
 * Generate basic block vectors for use with the SimPoint analysis tool.
 * SimPoint: https://cseweb.ucsd.edu/~calder/simpoint/
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Extended: .disas file output and plugin_flush() at exit.
 */

#include <stdio.h>
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

/* State machine for filtered recording */
enum plugin_state {
    STATE_DETECTING,    /* Waiting for target symbol */
    STATE_RECORDING    /* Recording only target function BBs */
};

/* Configuration (user-provided via plugin args) */
static char *target_func_name __attribute__((unused));      /* e.g. "ggml_gemv_q4_0_16x1_q8_0" */
static uint64_t target_func_size __attribute__((unused));   /* e.g. 0x30a (778 bytes) */

/* Detected/calculated at runtime */
static uint64_t func_start_vaddr __attribute__((unused));   /* Detected function entry address */
static uint64_t func_end_vaddr __attribute__((unused));     /* func_start_vaddr + target_func_size */

/* State tracking */
static enum plugin_state state __attribute__((unused));
static bool filter_enabled __attribute__((unused));         /* true when func_name specified */
static uint64_t detect_insn_count __attribute__((unused));  /* Timeout counter for detection */
static int symbol_match_count __attribute__((unused));      /* Number of symbol matches found */

#define MAX_DETECT_INSNS 100000     /* Timeout threshold */

/* Header tracking for disas output */
static bool disas_header_written __attribute__((unused));   /* Track if header written to disas */

static void free_bb(void *data)
{
    qemu_plugin_scoreboard_free(((Bb *)data)->count);
    g_free(data);
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

/*
 * Check if instruction is stack allocation (addi sp, sp, -imm)
 * Returns true for negative immediate stack growth.
 */
static bool is_stack_alloc_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        /* Compressed: C.ADDI16SP (c.addi16sp sp, nzimm) */
        uint16_t insn = (uint16_t)insn_raw;
        uint16_t funct3 = (insn >> 13) & 0x7;
        uint16_t rd = (insn >> 7) & 0x1F;
        uint16_t quadrant = insn & 0x3;
        /* C.ADDI16SP: funct3=3, rd=sp(2), quadrant=1 */
        return funct3 == 3 && rd == 2 && quadrant == 1;
    } else if (insn_size == 4) {
        /* 32-bit: ADDI rd=sp(2), rs1=sp(2), imm<0 */
        uint32_t opcode = insn_raw & 0x7F;
        uint32_t rd = (insn_raw >> 7) & 0x1F;
        uint32_t rs1 = (insn_raw >> 15) & 0x1F;
        uint32_t funct3 = (insn_raw >> 12) & 0x7;

        /* Sign-extend 12-bit immediate */
        int32_t imm = (int32_t)((insn_raw >> 20) & 0xFFF);
        if (imm & 0x800) imm |= 0xFFFFF000;

        return opcode == 0x13 &&       /* OP-IMM */
               funct3 == 0x0 &&        /* ADDI */
               rd == 2 &&              /* sp (x2) */
               rs1 == 2 &&             /* sp (x2) */
               imm < 0;                /* negative = stack growth */
    }
    return false;
}

/*
 * Check if instruction saves callee-saved register to stack.
 * Pattern: sd rs, offset(sp) or c.sd rs, offset(sp)
 */
static bool is_callee_save_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        /* Compressed: C.SDSP (c.sdsp rs2, offset(sp)) */
        uint16_t insn = (uint16_t)insn_raw;
        uint16_t quadrant = insn & 0x3;
        uint16_t funct3 = (insn >> 13) & 0x7;
        /* C.SDSP: quadrant=2 (C2), funct3=7, sp is implicit (no rs1 field) */
        return quadrant == 2 && funct3 == 7;
    } else if (insn_size == 4) {
        /* 32-bit: SD rs2, imm(rs1) */
        uint32_t opcode = insn_raw & 0x7F;
        uint32_t funct3 = (insn_raw >> 12) & 0x7;
        uint32_t rs1 = (insn_raw >> 15) & 0x1F;

        return opcode == 0x23 &&       /* STORE */
               funct3 == 0x3 &&        /* SD (64-bit) */
               rs1 == 2;               /* sp base */
    }
    return false;
}

/*
 * Detect RISC-V function prologue pattern.
 * Typical: addi sp, sp, -N followed by sd ra/s0, offset(sp)
 * Returns true if TB appears to be at function entry.
 */
__attribute__((unused))
static bool is_riscv_function_entry(struct qemu_plugin_tb *tb)
{
    size_t n_insns = qemu_plugin_tb_n_insns(tb);
    if (n_insns < 2) return false;

    /* Check first instruction for stack allocation */
    struct qemu_plugin_insn *insn0 = qemu_plugin_tb_get_insn(tb, 0);
    size_t size0 = qemu_plugin_insn_size(insn0);
    uint8_t data0[4] = {0};
    qemu_plugin_insn_data(insn0, data0, size0);
    uint32_t raw0 = (size0 == 2) ? *(uint16_t *)data0 : *(uint32_t *)data0;

    if (!is_stack_alloc_insn(raw0, size0)) {
        /* Some leaf functions skip stack allocation, check for sd anyway */
        struct qemu_plugin_insn *insn1 = qemu_plugin_tb_get_insn(tb, 1);
        size_t size1 = qemu_plugin_insn_size(insn1);
        uint8_t data1[4] = {0};
        qemu_plugin_insn_data(insn1, data1, size1);
        uint32_t raw1 = (size1 == 2) ? *(uint16_t *)data1 : *(uint32_t *)data1;

        return is_callee_save_insn(raw1, size1);
    }

    /* Stack allocation found, check second for callee-save */
    struct qemu_plugin_insn *insn1 = qemu_plugin_tb_get_insn(tb, 1);
    size_t size1 = qemu_plugin_insn_size(insn1);
    uint8_t data1[4] = {0};
    qemu_plugin_insn_data(insn1, data1, size1);
    uint32_t raw1 = (size1 == 2) ? *(uint16_t *)data1 : *(uint32_t *)data1;

    return is_callee_save_insn(raw1, size1) || n_insns > 2;
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    uint64_t n_insns = qemu_plugin_tb_n_insns(tb);
    uint64_t vaddr = qemu_plugin_tb_vaddr(tb);
    Bb *bb;

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
                fprintf(disas_file, "  0x%" PRIx64 ": %s\n", insn_vaddr, disas ? disas : "unknown");
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
            tokens[1] = NULL;  /* Prevent double-free */
        } else if (g_strcmp0(tokens[0], "func_size") == 0) {
            target_func_size = g_ascii_strtoull(tokens[1], NULL, 0);
        } else {
            fprintf(stderr, "option parsing failed: %s\n", opt);
            return -1;
        }
    }

    if (!filename) {
        fputs("outfile unspecified\n", stderr);
        return -1;
    }

    if (filter_enabled && !target_func_name) {
        fputs("func_name value required\n", stderr);
        return -1;
    }

    g_autofree gchar *disas_filename = g_strdup_printf("%s.disas", filename);
    disas_file = fopen(disas_filename, "w");
    if (!disas_file) {
        fprintf(stderr, "bbv: failed to open %s for writing — disassembly output disabled\n", disas_filename);
    }

    bbs = g_hash_table_new_full(g_int64_hash, g_int64_equal, NULL, free_bb);
    vcpus = qemu_plugin_scoreboard_new(sizeof(Vcpu));
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    qemu_plugin_register_vcpu_init_cb(id, vcpu_init);
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);

    return 0;
}
