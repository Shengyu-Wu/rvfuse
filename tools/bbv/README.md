# BBV Plugin for RISC-V

QEMU plugin for generating basic block vectors (BBV) for SimPoint analysis.
Extended with function-scoped recording capability.

## Standard Usage (Full-Program Recording)

```bash
qemu-riscv64 -L sysroot \
  -plugin libbbv.so,interval=10000,outfile=output/result \
  ./binary
```

Output files:
- `result.0.bb` - BBV data (interval-based)
- `result.disas` - Disassembly of all recorded basic blocks

## Function-Scoped Recording

To record only basic blocks within a specific target function:

### 1. Build target binary with symbols

Ensure the binary is not stripped (no `install/strip` in build).

### 2. Get function size from nm

```bash
nm -D -S lib/libggml-cpu.so | grep <function_name>
# Example: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
```

The second column is the function size (e.g., `0x30a`).

### 3. Run with function filter

```bash
qemu-riscv64 -L sysroot \
  -plugin libbbv.so,func_name=<symbol>,func_size=<hex>,outfile=<path> \
  ./binary
```

**Parameters:**
- `func_name`: Target function symbol name (required for filtering)
- `func_size`: Function size in hex from nm output (required)
- `interval`: BBV output interval (default: 100000000)
- `outfile`: Output file prefix (required)

**Example:**
```bash
qemu-riscv64 -L sysroot \
  -plugin libbbv.so,func_name=ggml_gemv_q4_0_16x1_q8_0,func_size=0x30a,interval=10000,outfile=output/gemv \
  ./llama-bench -m model.gguf -p 512
```

## Behavior

- **Without `func_name`**: Original behavior (full-program recording)
- **With `func_name`**: Only records BBs within target function address range
- Detection uses symbol lookup + RISC-V prologue verification
- Timeout fallback after 100,000 instructions if prologue detection fails

## Output Format

### .bb file
SimPoint-compatible format:
```
T:<bb_index>:<count> ...
```

### .disas file
- Header: Function info (name, size, detected range)
- Basic blocks: Address, instruction count, disassembly

## Limitations

- Requires unstripped binary with symbols
- Function must be executed during program run
- Dynamic linking: uses runtime symbol detection