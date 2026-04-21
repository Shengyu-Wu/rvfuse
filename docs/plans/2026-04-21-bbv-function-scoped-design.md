# BBV Plugin Function-Scoped Recording Design

**Date**: 2026-04-21
**Author**: Claude (brainstorming session)
**Status**: Approved

## Problem Statement

The current BBV plugin records all basic blocks from program start to finish, causing significant slowdown for long-running programs. The goal is to upgrade the plugin to record only basic blocks within a specified target function, dramatically reducing overhead.

**Constraints**:
- No modifications to QEMU source code (plugin-only changes)
- Must work with dynamically linked binaries (ASLR address randomization)
- Must be applicable to any application (not just llama.cpp)

## Solution Overview

Implement a **symbol-triggered + real-time filtering** approach:

1. **Detection phase**: Wait for target function symbol to appear
2. **Trigger**: When symbol matches + RISC-V prologue detected, calculate address range
3. **Recording phase**: Only record BBs within `[func_start, func_start + func_size]`

## Architecture

### State Machine

```
DETECTING ──→ TRIGGERED ──→ RECORDING
```

| State | Behavior |
|--------|----------|
| DETECTING | Check each TB for target symbol, no BB recording |
| TRIGGERED | Found symbol with prologue, compute address range |
| RECORDING | Only record BBs within target function address range |

### Core Data Structures

```c
/* State machine */
enum {
    STATE_DETECTING,    /* Waiting for target symbol */
    STATE_TRIGGERED,    /* Found symbol, computing range */
    STATE_RECORDING     /* Only recording target function BBs */
} state;

/* Configuration (user-provided) */
static char *target_func_name;     /* e.g. "ggml_gemv_q4_0_16x1_q8_0" */
static uint64_t target_func_size;  /* e.g. 0x30a (778 bytes) */

/* Detected at runtime */
static uint64_t func_start_vaddr;  /* Detected function entry address */
static uint64_t func_end_vaddr;    /* func_start_vaddr + target_func_size */

/* Modified hash table (filtered) */
static GHashTable *bbs_in_range;   /* Only BBs within target function */
static bool filter_enabled;        /* true when func_name specified */
```

## Parameters

**New plugin options**:

```
-plugin libbbv.so,func_name=<symbol>,func_size=<hex>,outfile=<path>
```

| Parameter | Description | Example |
|-----------|-------------|---------|
| `func_name` | Target function symbol name | `ggml_gemv_q4_0_16x1_q8_0` |
| `func_size` | Function size in hex (from nm) | `0x30a` |
| `outfile` | Output file prefix (existing) | `output/result` |
| `interval` | Interval for output (existing) | `10000` |

**Backward compatibility**: If `func_name` not specified, plugin behaves identically to original (full-program recording).

## Address Detection Strategy

### Step 1: Build with Symbols

Modify build process to preserve symbols:

```bash
# In build.sh, change:
#   ninja install/strip  →  ninja install
```

### Step 2: Get Static Function Info

```bash
nm -D -S lib/libggml-cpu.so | grep <function_name>
# Output: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
#         address            size              type  name
```

### Step 3: Runtime Base Address Detection

In `vcpu_tb_trans` callback:

```c
for (size_t i = 0; i < qemu_plugin_tb_n_insns(tb); i++) {
    struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
    const char *sym = qemu_plugin_insn_symbol(insn);

    if (sym && g_strcmp0(sym, target_func_name) == 0) {
        if (is_riscv_function_entry(tb)) {
            /* Found function entry! */
            func_start_vaddr = qemu_plugin_tb_vaddr(tb);
            func_end_vaddr = func_start_vaddr + target_func_size;
            state = STATE_RECORDING;
            break;
        }
    }
}
```

### Step 4: RISC-V Prologue Detection

Verify TB is at function entry by checking prologue pattern:

```asm
# Typical RISC-V function prologue
addi    sp, sp, -N      # Stack allocation (negative immediate)
sd      ra, offset(sp)  # Save return address
sd      s0, offset(sp)  # Save callee-saved registers
```

Detection logic checks:
1. First instruction is `addi sp, sp, -imm` (stack allocation)
2. Following instruction saves callee-saved register to stack

This ensures we detect the actual function entry, not an internal position.

## BB Recording Modification

**Original behavior**: Instrument all TBs, store in global `bbs` hash table.

**New behavior (filtering mode)**:

```c
static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb)
{
    uint64_t vaddr = qemu_plugin_tb_vaddr(tb);

    if (filter_enabled) {
        switch (state) {
        case STATE_DETECTING:
            detect_symbol(tb);
            return;  /* No instrumentation */

        case STATE_RECORDING:
            if (vaddr >= func_start_vaddr && vaddr < func_end_vaddr) {
                instrument_bb(tb, vaddr);  /* Only within range */
            }
            return;
        }
    }

    /* Original mode: no filtering */
    instrument_bb(tb, vaddr);
}
```

## Error Handling

### Function Never Executed

```c
/* At plugin_exit */
if (filter_enabled && state == STATE_DETECTING) {
    qemu_plugin_outs("BBV WARNING: Target function never executed.\n");
    /* Write empty output with explanation */
}
```

### Prologue Detection Timeout

After N instructions without finding prologue, fallback to first symbol match:

```c
#define MAX_DETECT_INSNS 100000
static uint64_t detect_insn_count;

if (detect_insn_count > MAX_DETECT_INSNS) {
    /* Use first symbol match as fallback */
    state = STATE_RECORDING;
}
```

### Multiple Symbol Matches

Log each match but only use the one with valid prologue:

```c
symbol_match_count++;
if (!is_riscv_function_entry(tb)) {
    qemu_plugin_outs("BBV: Symbol matched but not prologue\n");
    return;  /* Keep detecting */
}
```

## Usage Example

```bash
# 1. Build with symbols
./build.sh --no-strip

# 2. Get function size
nm -D -S output/llama.cpp/lib/libggml-cpu.so | grep gemv_q4_0_16x1_q8_0
# 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0

# 3. Run BBV with function filter
third_party/qemu/build/qemu-riscv64 -L output/llama.cpp/sysroot \
  -plugin tools/bbv/libbbv.so,\
func_name=ggml_gemv_q4_0_16x1_q8_0,\
func_size=0x30a,\
interval=10000,\
outfile=output/llama-func-bbv \
  output/llama.cpp/bin/llama-cli -m model.gguf -p "Hello"

# 4. Output files
#   output/llama-func-bbv.0.bb     - BBV data for function only
#   output/llama-func-bbv.disas    - Disassembly of function BBs
```

## Implementation Checklist

1. [ ] Add state machine enum and variables
2. [ ] Parse new parameters (`func_name`, `func_size`)
3. [ ] Implement RISC-V prologue detection
4. [ ] Modify `vcpu_tb_trans` for state-based filtering
5. [ ] Modify hash table to `bbs_in_range`
6. [ ] Add error handling (timeout, no match, etc.)
7. [ ] Add disas output header with detection info
8. [ ] Test with llama.cpp target function
9. [ ] Verify backward compatibility (no func_name = original behavior)

## Performance Expectations

| Metric | Original | Function-Scoped |
|--------|----------|-----------------|
| Memory | O(all BBs) | O(target function BBs only) |
| Detection overhead | 0 | O(1) per TB (symbol check) until trigger |
| Recording overhead | O(all TBs) | O(target function TBs only) |
| Output file size | Full program | Target function only |

**Expected improvement**: For a function that represents 5% of execution time, overhead should reduce by ~95% after detection phase.