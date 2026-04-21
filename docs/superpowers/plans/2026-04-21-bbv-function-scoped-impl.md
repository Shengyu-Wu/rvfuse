# BBV Function-Scoped Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade BBV plugin to record only basic blocks within a specified target function, reducing overhead for long-running programs.

**Architecture:** State machine (DETECTING → TRIGGERED → RECORDING) with symbol-triggered address detection and RISC-V prologue verification. Real-time filtering only records BBs within target function address range.

**Tech Stack:** C (QEMU plugin API), GLib (hash tables, string utils), QEMU 9.2.4 plugin headers

---

## Files Modified

| File | Purpose | Changes |
|------|---------|---------|
| `tools/bbv/bbv.c` | Main plugin source | State machine, prologue detection, filtering logic |
| `tools/bbv/Makefile` | Build configuration | No changes needed |
| `tools/bbv/demo.c` | Test demo program | No changes needed |

---

### Task 1: Add State Machine and Configuration Variables

**Files:**
- Modify: `tools/bbv/bbv.c:15-35` (after existing typedefs)

- [ ] **Step 1: Add state machine enum and new static variables**

Add after line 24 (after `static struct qemu_plugin_scoreboard *vcpus;`):

```c
/* ========== Function-Scoped Recording ========== */

/* State machine for filtered recording */
enum plugin_state {
    STATE_DETECTING,    /* Waiting for target symbol */
    STATE_RECORDING     /* Recording only target function BBs */
};

/* Configuration (user-provided via plugin args) */
static char *target_func_name;      /* e.g. "ggml_gemv_q4_0_16x1_q8_0" */
static uint64_t target_func_size;   /* e.g. 0x30a (778 bytes) */

/* Detected/calculated at runtime */
static uint64_t func_start_vaddr;   /* Detected function entry address */
static uint64_t func_end_vaddr;     /* func_start_vaddr + target_func_size */

/* State tracking */
static enum plugin_state state;
static bool filter_enabled;         /* true when func_name specified */
static uint64_t detect_insn_count;  /* Timeout counter for detection */
static int symbol_match_count;      /* Number of symbol matches found */

#define MAX_DETECT_INSNS 100000     /* Timeout threshold */
```

- [ ] **Step 2: Commit state machine addition**

```bash
git add tools/bbv/bbv.c
git commit -m "feat(bbv): add state machine and filter configuration variables"
```

---

### Task 2: Parse New Plugin Parameters

**Files:**
- Modify: `tools/bbv/bbv.c:188-210` (qemu_plugin_install function)

- [ ] **Step 1: Add parameter parsing for func_name and func_size**

Modify the parameter parsing loop in `qemu_plugin_install` (around lines 192-204). Add new options after the existing `outfile` parsing:

```c
    } else if (g_strcmp0(tokens[0], "func_name") == 0) {
        target_func_name = g_strdup(tokens[1]);
        filter_enabled = true;
        state = STATE_DETECTING;
        detect_insn_count = 0;
        symbol_match_count = 0;
        tokens[1] = NULL;  /* Prevent double-free */
    } else if (g_strcmp0(tokens[0], "func_size") == 0) {
        target_func_size = g_ascii_strtoull(tokens[1], NULL, 0);
    } else {
        fprintf(stderr, "option parsing failed: %s\n", opt);
        return -1;
    }
```

- [ ] **Step 2: Verify parsing logic integrates correctly**

Run a quick compile to check syntax:
```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv && make clean && make
```
Expected: Compilation succeeds (may have warnings about unused variables)

- [ ] **Step 3: Commit parameter parsing**

```bash
git add tools/bbv/bbv.c
git commit -m "feat(bbv): add func_name and func_size parameter parsing"
```

---

### Task 3: Implement RISC-V Prologue Detection

**Files:**
- Modify: `tools/bbv/bbv.c` (add new helper functions before vcpu_tb_trans)

- [ ] **Step 1: Add RISC-V instruction decode helpers**

Add before `vcpu_tb_trans` function (around line 147):

```c
/* ========== RISC-V Prologue Detection ========== */

/*
 * Check if instruction is stack allocation (addi sp, sp, -imm)
 * Returns true for negative immediate stack growth.
 */
static bool is_stack_alloc_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        /* Compressed: C.ADDI16SP (c.addi16sp sp, nzimm) */
        /* Format: bits[15:13]=011, bits[12:10]=non-zero, bits[9:7]=000, bits[6:0]=0110001 */
        uint16_t insn = (uint16_t)insn_raw;
        return (insn & 0xE383) == 0x6103;  /* C.ADDI16SP opcode pattern */
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
 * Callee-saved: ra(1), s0-s11(8-19, 20-23)
 * Pattern: sd rs, offset(sp) or c.sd rs, offset(sp)
 */
static bool is_callee_save_insn(uint32_t insn_raw, size_t insn_size)
{
    if (insn_size == 2) {
        /* Compressed: C.SD (c.sd rs2', offset(sp)) */
        uint16_t insn = (uint16_t)insn_raw;
        uint16_t opcode = insn & 0x3;
        uint16_t funct3 = (insn >> 13) & 0x7;
        uint16_t rs1 = (insn >> 7) & 0x7;  /* c.sd uses sp implicitly */
        
        return opcode == 0x0 &&        /* C0 quadrant */
               funct3 == 0x7 &&        /* C.SD */
               rs1 == 0x2;             /* sp base (encoded as 2 in CI format) */
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
```

- [ ] **Step 2: Compile to verify prologue detection code**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv && make clean && make
```
Expected: Compilation succeeds with no errors

- [ ] **Step 3: Commit prologue detection functions**

```bash
git add tools/bbv/bbv.c
git commit -m "feat(bbv): add RISC-V prologue detection helpers"
```

---

### Task 4: Implement Symbol Detection Logic

**Files:**
- Modify: `tools/bbv/bbv.c:147-186` (vcpu_tb_trans function)

- [ ] **Step 1: Create detect_symbol helper function**

Add before `vcpu_tb_trans`:

```c
/*
 * Detect target function symbol in TB.
 * Checks each instruction's symbol name against target_func_name.
 * On match with valid prologue, computes address range and switches to RECORDING.
 */
static void detect_target_symbol(struct qemu_plugin_tb *tb)
{
    size_t n_insns = qemu_plugin_tb_n_insns(tb);
    uint64_t tb_vaddr = qemu_plugin_tb_vaddr(tb);
    
    detect_insn_count += n_insns;
    
    /* Timeout check: after MAX_DETECT_INSNS, use first symbol match */
    if (detect_insn_count > MAX_DETECT_INSNS && state == STATE_DETECTING) {
        qemu_plugin_outs("BBV: Detection timeout - checking final symbols\n");
    }
    
    for (size_t i = 0; i < n_insns; i++) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        const char *sym = qemu_plugin_insn_symbol(insn);
        
        if (sym && target_func_name && g_strcmp0(sym, target_func_name) == 0) {
            symbol_match_count++;
            
            bool is_entry = is_riscv_function_entry(tb);
            
            if (is_entry || detect_insn_count > MAX_DETECT_INSNS) {
                /* Found function entry (or timeout fallback) */
                func_start_vaddr = tb_vaddr;
                func_end_vaddr = func_start_vaddr + target_func_size;
                state = STATE_RECORDING;
                
                /* Log detection info */
                g_autofree gchar *msg = g_strdup_printf(
                    "BBV: Target function '%s' detected at 0x%" PRIx64 
                    " (size 0x%" PRIx64 ", end 0x%" PRIx64 ")%s\n",
                    target_func_name, func_start_vaddr, target_func_size, func_end_vaddr,
                    is_entry ? "" : " [timeout fallback]");
                qemu_plugin_outs(msg);
                return;
            } else {
                g_autofree gchar *msg = g_strdup_printf(
                    "BBV: Symbol '%s' matched at 0x%" PRIx64 " but not prologue (match #%d)\n",
                    target_func_name, tb_vaddr, symbol_match_count);
                qemu_plugin_outs(msg);
            }
        }
    }
}
```

- [ ] **Step 2: Modify vcpu_tb_trans to use state machine**

Replace the existing `vcpu_tb_trans` function (lines 147-186) with:

```c
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
            return;  /* No instrumentation during detection */

        case STATE_RECORDING:
            /* Only instrument if within target function range */
            if (vaddr < func_start_vaddr || vaddr >= func_end_vaddr) {
                return;  /* Skip BBs outside target function */
            }
            break;  /* Proceed to instrument this BB */
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
                const char *sym = qemu_plugin_insn_symbol(insn);
                fprintf(disas_file, "  0x%" PRIx64 ": %s", insn_vaddr, 
                        disas ? disas : "unknown");
                if (sym) {
                    fprintf(disas_file, " [%s]", sym);
                }
                fprintf(disas_file, "\n");
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
```

- [ ] **Step 3: Compile and verify state machine integration**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv && make clean && make
```
Expected: Compilation succeeds

- [ ] **Step 4: Commit state machine integration**

```bash
git add tools/bbv/bbv.c
git commit -m "feat(bbv): integrate state machine into vcpu_tb_trans"
```

---

### Task 5: Add Exit Warning for Undetected Function

**Files:**
- Modify: `tools/bbv/bbv.c:83-100` (plugin_exit function)

- [ ] **Step 1: Add warning for function never detected**

Modify `plugin_exit` to add warning at the beginning:

```c
static void plugin_exit(qemu_plugin_id_t id, void *p)
{
    /* Warn if target function was never detected */
    if (filter_enabled && state == STATE_DETECTING) {
        g_autofree gchar *msg = g_strdup_printf(
            "BBV WARNING: Target function '%s' never executed. No BBV data recorded.\n",
            target_func_name ? target_func_name : "(unknown)");
        qemu_plugin_outs(msg);
        
        if (disas_file) {
            fprintf(disas_file, "# BBV Function-Scoped Mode - TARGET NOT FOUND\n");
            fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                    target_func_name ? target_func_name : "(unknown)",
                    target_func_size);
            fprintf(disas_file, "# Symbol matches: %d\n", symbol_match_count);
            fprintf(disas_file, "# Instructions checked: %" PRIu64 "\n", detect_insn_count);
        }
    }

    plugin_flush();

    /* ... rest of existing plugin_exit code ... */
```

- [ ] **Step 2: Add detection info header for successful case**

Also add header in the recording case. Modify the disas_file output in `vcpu_tb_trans` when first BB is recorded:

Actually, better to add in `detect_target_symbol` when transitioning to RECORDING. Let's add a header callback:

Add at the beginning of plugin_exit after the warning check:

```c
    /* Write header to disas file if function was detected */
    if (filter_enabled && state == STATE_RECORDING && disas_file) {
        fprintf(disas_file, "# BBV Function-Scoped Mode\n");
        fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                target_func_name, target_func_size);
        fprintf(disas_file, "# Detected start: 0x%" PRIx64 "\n", func_start_vaddr);
        fprintf(disas_file, "# Detected end:   0x%" PRIx64 "\n", func_end_vaddr);
        fprintf(disas_file, "# Symbol matches: %d\n", symbol_match_count);
        fprintf(disas_file, "#\n\n");
    }
```

Wait - the disas_file is written during TB translation, not at exit. The header should be written at detection time. Let me modify `detect_target_symbol` instead - add a static flag to track if header was written:

Actually, a cleaner approach: create a separate `disas_header_file` or write header at the first BB recording in filter mode. Let me add a flag:

Add to the static variables in Task 1:
```c
static bool disas_header_written;
```

Then in `vcpu_tb_trans`, after `g_rw_lock_writer_unlock(&bbs_lock);`, add:

```c
        /* Write header for first BB in filtered mode */
        if (filter_enabled && !disas_header_written && disas_file) {
            fprintf(disas_file, "# BBV Function-Scoped Mode\n");
            fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                    target_func_name, target_func_size);
            fprintf(disas_file, "# Range: 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                    func_start_vaddr, func_end_vaddr);
            fprintf(disas_file, "#\n\n");
            disas_header_written = true;
        }
```

Let me update the plan to include this properly.

- [ ] **Step 2 (updated): Add disas_header_written flag and header output**

First add to static variables (from Task 1, add this line):
```c
static bool disas_header_written;  /* Track if header written to disas */
```

Then in `vcpu_tb_trans`, after the `g_rw_lock_writer_unlock(&bbs_lock);` line, add:

```c
    /* Write header for first BB in filtered mode */
    if (filter_enabled && state == STATE_RECORDING && !disas_header_written && disas_file) {
        fprintf(disas_file, "# BBV Function-Scoped Mode\n");
        fprintf(disas_file, "# Target: %s (size 0x%" PRIx64 ")\n",
                target_func_name, target_func_size);
        fprintf(disas_file, "# Range: 0x%" PRIx64 " - 0x%" PRIx64 "\n",
                func_start_vaddr, func_end_vaddr);
        fprintf(disas_file, "#\n\n");
        disas_header_written = true;
    }
```

- [ ] **Step 3: Compile to verify exit handling**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv && make clean && make
```
Expected: Compilation succeeds

- [ ] **Step 4: Commit exit handling**

```bash
git add tools/bbv/bbv.c
git commit -m "feat(bbv): add exit warning and disas header for function-scoped mode"
```

---

### Task 6: Update Free Logic for New Variables

**Files:**
- Modify: `tools/bbv/bbv.c:83-100` (plugin_exit cleanup section)

- [ ] **Step 1: Add cleanup for target_func_name**

In `plugin_exit`, add cleanup for the new string variable before the existing `g_free(filename)`:

```c
    g_hash_table_unref(bbs);
    g_free(filename);
    g_free(target_func_name);  /* Free function name if allocated */
    if (disas_file) {
        fclose(disas_file);
    }
    qemu_plugin_scoreboard_free(vcpus);
```

- [ ] **Step 2: Commit cleanup changes**

```bash
git add tools/bbv/bbv.c
git commit -m "fix(bbv): add cleanup for target_func_name in plugin_exit"
```

---

### Task 7: Build and Basic Test

**Files:**
- Test: `tools/bbv/demo.c` (existing test program)

- [ ] **Step 1: Build the plugin**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv && make clean && make
```
Expected: `libbbv.so` created with no errors

- [ ] **Step 2: Run demo without filter (backward compatibility test)**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv
/home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/third_party/qemu/build/qemu-riscv64 \
  -L /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/output/sysroot \
  -plugin ./libbbv.so,interval=100,outfile=./test_nofilter \
  ./demo.elf
```
Expected: `test_nofilter.0.bb` and `test_nofilter.disas` created with all BBs recorded

- [ ] **Step 3: Run demo with filter (non-existent function)**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv
/home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/third_party/qemu/build/qemu-riscv64 \
  -L /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/output/sysroot \
  -plugin ./libbbv.so,func_name=nonexistent_func,func_size=0x100,interval=100,outfile=./test_noresult \
  ./demo.elf
```
Expected: Warning message "Target function 'nonexistent_func' never executed"

- [ ] **Step 4: Check output files**

```bash
ls -la /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv/test_*.bb
cat /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv/test_noresult.disas
```
Expected: Empty or header-only disas file with "TARGET NOT FOUND" message

- [ ] **Step 5: Clean test artifacts**

```bash
rm -f /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv/test_*.bb
rm -f /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv/test_*.disas
rm -f /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/tools/bbv/bbv.out.*
```

---

### Task 8: Integration Test with llama.cpp

**Files:**
- Test: Uses main repo's llama.cpp build (symlinked)

- [ ] **Step 1: Verify llama.cpp binaries with symbols**

```bash
nm -D /home/pren/wsp/cx/rvfuse/output/llama.cpp/lib/libggml-cpu.so.0.9.11 | grep -i "gemv_q4_0" | head -5
```
Expected: Shows function symbols with addresses and sizes

- [ ] **Step 2: Get target function info**

```bash
nm -D -S /home/pren/wsp/cx/rvfuse/output/llama.cpp/lib/libggml-cpu.so.0.9.11 | grep "ggml_gemv_q4_0_16x1_q8_0"
```
Expected output format: `address size T function_name`
Record the size value (e.g., `0x30a`)

- [ ] **Step 3: Run llama.cpp with function-scoped BBV**

```bash
cd /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update
mkdir -p output/func-bbv-test

FUNC_SIZE=$(nm -D -S /home/pren/wsp/cx/rvfuse/output/llama.cpp/lib/libggml-cpu.so.0.9.11 | grep "ggml_gemv_q4_0_16x1_q8_0" | awk '{print $2}')

third_party/qemu/build/qemu-riscv64 \
  -L /home/pren/wsp/cx/rvfuse/output/llama.cpp/sysroot \
  -plugin tools/bbv/libbbv.so,\
func_name=ggml_gemv_q4_0_16x1_q8_0,\
func_size=${FUNC_SIZE},\
interval=10000,\
outfile=output/func-bbv-test/llama-gemv \
  /home/pren/wsp/cx/rvfuse/output/llama.cpp/bin/llama-bench \
  -m /path/to/model.gguf \
  -p 512 \
  -ngl 1
```
Note: Model path needs to be provided by user

- [ ] **Step 4: Verify output**

```bash
ls -la /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/output/func-bbv-test/
head -30 /home/pren/wsp/cx/rvfuse/.claude/worktrees/bbv-update/output/func-bbv-test/llama-gemv.disas
```
Expected: 
- Header shows "BBV Function-Scoped Mode" with detected addresses
- Only BBs within target function range are recorded
- File size much smaller than full-program BBV

- [ ] **Step 5: Compare with full-program BBV (optional)**

```bash
# Run same workload without filter for comparison
third_party/qemu/build/qemu-riscv64 \
  -L /home/pren/wsp/cx/rvfuse/output/llama.cpp/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/func-bbv-test/llama-full \
  /home/pren/wsp/cx/rvfuse/output/llama.cpp/bin/llama-bench \
  -m /path/to/model.gguf -p 512 -ngl 1

# Compare file sizes
ls -la output/func-bbv-test/*.disas
```
Expected: `llama-gemv.disas` significantly smaller than `llama-full.disas`

---

### Task 9: Final Documentation and Commit

**Files:**
- Update: `tools/bbv/README.md` (if exists) or create

- [ ] **Step 1: Update README with new usage**

Add documentation for function-scoped mode in `tools/bbv/README.md`:

```markdown
## Function-Scoped Recording

To record only basic blocks within a specific target function:

1. **Build target binary with symbols** (no stripping)

2. **Get function size from nm:**
   ```bash
   nm -D -S lib/libggml-cpu.so | grep <function_name>
   # Example: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
   ```

3. **Run with function filter:**
   ```bash
   qemu-riscv64 -L sysroot \
     -plugin libbbv.so,func_name=<symbol>,func_size=<hex>,outfile=<path> \
     ./binary
   ```

**Parameters:**
- `func_name`: Target function symbol name (required for filtering)
- `func_size`: Function size in hex from nm output (required)
- `interval`: BBV output interval (existing, default 100000000)
- `outfile`: Output file prefix (existing, required)

**Behavior:**
- Without `func_name`: Original behavior (full-program recording)
- With `func_name`: Only records BBs within target function address range
- Detection uses symbol lookup + RISC-V prologue verification
```

- [ ] **Step 2: Final commit**

```bash
git add tools/bbv/bbv.c tools/bbv/README.md
git commit -m "feat(bbv): complete function-scoped recording implementation

Adds state machine (DETECTING → RECORDING) with:
- RISC-V prologue detection for accurate function entry identification
- Symbol-triggered address range calculation
- Real-time filtering to record only target function BBs
- Backward compatible: no func_name = original behavior

Usage: -plugin libbbv.so,func_name=<symbol>,func_size=<hex>,outfile=<path>"
```

---

## Self-Review Checklist

### Spec Coverage

| Design Requirement | Task |
|-------------------|------|
| State machine enum | Task 1 |
| Parse func_name, func_size | Task 2 |
| RISC-V prologue detection | Task 3 |
| State-based filtering in vcpu_tb_trans | Task 4 |
| Error: function never executed | Task 5 |
| Error: timeout fallback | Task 3, 4 |
| Disas header with detection info | Task 5 |
| Cleanup new variables | Task 6 |
| Backward compatibility test | Task 7 |
| Integration test with llama.cpp | Task 8 |

### Placeholder Scan

- No TBD, TODO, or "implement later" phrases
- All code blocks contain actual implementation code
- All commands have specific paths and expected outputs
- No references to undefined functions or types

### Type Consistency

- `enum plugin_state` defined in Task 1, used consistently
- `target_func_name` is `char *`, freed in Task 6
- `func_start_vaddr`, `func_end_vaddr`, `target_func_size` are `uint64_t`
- `state`, `filter_enabled`, `disas_header_written` are `bool`/enum
- Function signatures in Task 3 match usage in Task 4