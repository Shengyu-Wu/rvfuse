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

### Method 1: Symbol-Based Detection (Main Program)

For statically linked binaries or main program symbols:

#### 1. Build target binary with symbols

Ensure the binary is not stripped (no `install/strip` in build).

#### 2. Get function size from nm

```bash
nm -D -S lib/libggml-cpu.so | grep <function_name>
# Example: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
```

The second column is the function size (e.g., `0x30a`).

#### 3. Run with function filter

```bash
qemu-riscv64 -L sysroot \
  -plugin libbbv.so,func_name=<symbol>,func_size=<hex>,outfile=<path> \
  ./binary
```

### Method 2: Direct Address Mode (Shared Libraries)

For shared library functions where symbol detection doesn't work in QEMU user-mode:

#### 1. Run full-program BBV to find function address

```bash
# First run with ASLR disabled for consistent addresses
setarch x86_64 -R \
  qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib \
  -plugin libbbv.so,interval=100000,outfile=output/full \
  ./binary

# Find target function in disas output (look for characteristic instructions)
grep -A5 "vsetivli.*zero,16,e32" output/full.disas
# Output: BB N (vaddr: 0x7ffff66dedee, ...) shows function prologue
```

#### 2. Get function size from nm

```bash
nm -D -S lib/libggml-cpu.so | grep <function_name>
# Example: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
```

#### 3. Calculate decimal address

```python3
addr = 0x7ffff66dedee  # from step 1
size = 0x30a           # from step 2
print(f"Address: {addr}, Size: {size}")
```

#### 4. Run with direct address

```bash
setarch x86_64 -R \
  qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib \
  -plugin libbbv.so,func_addr=<decimal_addr>,func_size=<decimal_size>,interval=1000,outfile=output/gemv \
  ./binary
```

**Example with llama.cpp:**
```bash
# Step 1: Find gemv function address
setarch x86_64 -R timeout 60 \
  qemu-riscv64 -L llama-sysroot -E LD_LIBRARY_PATH=llama-lib -cpu max \
  -plugin libbbv.so,interval=100000,outfile=output/qwen-full \
  llama-bin/llama-completion -m model.gguf -p "Hello" -n 10

grep -A5 "addi.*sp,sp,-144" output/qwen-full.disas | grep vsetivli
# Found: 0x7ffff66dedee

# Step 2: Run function-scoped BBV
setarch x86_64 -R timeout 60 \
  qemu-riscv64 -L llama-sysroot -E LD_LIBRARY_PATH=llama-lib -cpu max \
  -plugin libbbv.so,func_addr=140737327787502,func_size=778,interval=1000,outfile=output/gemv \
  llama-bin/llama-completion -m model.gguf -p "Hello" -n 20
```

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `func_name` | Symbol name (main program) | `ggml_gemv_q4_0_16x1_q8_0` |
| `func_addr` | Direct runtime address (decimal) | `140737327787502` |
| `func_size` | Function size (hex or decimal) | `0x30a` or `778` |
| `lib_name` | Library name for detection | `libggml-cpu` |
| `func_offset` | Static offset from nm | `0xac9a4` |
| `interval` | BBV output interval | `10000` |
| `outfile` | Output file prefix | `output/result` |

## Behavior

- **Without filtering**: Original behavior (full-program recording)
- **func_name mode**: Symbol lookup + RISC-V prologue verification
- **func_addr mode**: Direct address range filtering (requires ASLR disabled)
- **lib_name+func_offset mode**: Library base detection (experimental)

## Output Format

### .bb file
SimPoint-compatible format:
```
T:<bb_index>:<count> ...
```

### .disas file
- Header: Function info (mode, address, size, range)
- Basic blocks: Address, instruction count, disassembly

## Limitations

- **Symbol-based detection**: Only works for main program symbols (not shared libraries)
- **Direct address mode**: Requires ASLR disabled (`setarch x86_64 -R`) for consistent addresses
- **Address discovery**: Must run full-program BBV first to find function address
- Dynamic linking symbol resolution: QEMU user-mode doesn't resolve shared library symbols via `qemu_plugin_insn_symbol()`