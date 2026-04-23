# BBV Plugin Implementation Details

This file contains technical implementation details for the custom BBV plugin.
For usage instructions, see SKILL.md.

## Table of Contents

1. [Data Structures](#data-structures)
2. [Key Callbacks](#key-callbacks)
3. [Syscall Detection Flow](#syscall-detection-flow)
4. [Syscall Numbers (RISC-V)](#syscall-numbers-risc-v)

## Data Structures

```c
// Standard BB tracking
typedef struct Bb {
    uint64_t vaddr;              // Basic block virtual address
    struct qemu_plugin_scoreboard *count;  // Per-vCPU execution count
    unsigned int index;          // BB index for output
} Bb;

typedef struct Vcpu {
    uint64_t count;              // Instruction counter for interval
    FILE *file;                  // Output file handle
} Vcpu;

// Function-scoped recording configuration
static char *lib_name;           // Library name for syscall detection
static uint64_t func_offset;     // Static offset from nm
static uint64_t target_func_size;
static uint64_t func_addr;       // Direct runtime address

// Detected at runtime
static uint64_t func_start_vaddr;
static uint64_t func_end_vaddr;
static enum plugin_state state;  // STATE_DETECTING or STATE_RECORDING
```

## Key Callbacks

| Callback | Purpose |
|----------|---------|
| `vcpu_init` | Open per-vCPU output file |
| `vcpu_tb_trans` | Register new BB, record disassembly, filter by function range |
| `vcpu_interval_exec` | Flush counts when interval reached |
| `vcpu_syscall` | Intercept openat/mmap for library detection |
| `vcpu_syscall_ret` | Process syscall return values, compute function address |
| `plugin_exit` | Final flush and cleanup |

## Syscall Detection Flow

For `lib_name` + `func_offset` mode:

1. **openat syscall entry**: Check if pathname contains target library name
2. **openat syscall return**: Track file descriptor if target library
3. **mmap syscall entry**: Save prot flags, fd, offset arguments
4. **mmap syscall return**: If fd matches tracked library fd and `PROT_EXEC`:
   - Compute library base address from mmap return value
   - Calculate `func_start_vaddr = base + vaddr_offset + func_offset`
   - Transition to `STATE_RECORDING`
5. **TB translation**: Only register BBs within `[func_start, func_start + size]`

### Address Calculation

```c
// From mmap return value (guest a0 in syscall ret callback):
uint64_t mmap_ret_addr = a1;  // Return value is in guest a0 = callback a1

// From ELF p_vaddr and file_offset:
// The text segment is mapped at: mmap_ret_addr
// Function address = mmap_ret_addr + (nm_offset - p_vaddr + file_offset)
// Simplified: func_addr = elf_base + nm_offset
```

## Syscall Numbers (RISC-V)

| Syscall | Number | Purpose |
|---------|--------|---------|
| `openat` | 56 | Open file by path (library loading) |
| `mmap` | 222 | Memory mapping (library text segment) |

**Note**: These numbers are RISC-V specific. The plugin will not work on other architectures.

## Interval Behavior

The plugin counts instructions and triggers output when count reaches `interval`:

```c
// After each translation block execution:
qemu_plugin_register_vcpu_tb_exec_inline_per_vcpu(
    tb, QEMU_PLUGIN_INLINE_ADD_U64, count_u64(), n_insns);

// When count >= interval, flush and reset:
qemu_plugin_register_vcpu_tb_exec_cond_cb(
    tb, vcpu_interval_exec, QEMU_PLUGIN_CB_NO_REGS,
    QEMU_PLUGIN_COND_GE, count_u64(), interval, NULL);
```

At program exit, `plugin_flush()` writes any remaining counts.