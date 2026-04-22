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

### Method 2: Syscall-Based Library Detection (Shared Libraries)

**Recommended for shared library profiling.** Automatically detects library
loading via `openat`/`mmap` syscall interception — no preliminary full-program
run needed.

#### 1. Get function offset and size from nm

```bash
nm -D -S lib/libggml-cpu.so | grep <function_name>
# Example: 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
#          ^offset                        ^size
```

#### 2. Run with library detection

```bash
qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib -cpu max \
  -plugin libbbv.so,lib_name=libggml-cpu,func_offset=0xac9a4,func_size=0x30a,interval=1000,outfile=output/gemv \
  ./llama-completion -m model.gguf -p "Hello" -n 20
```

**How it works:**
1. Intercepts `openat` syscalls to detect when the target library is opened
2. Tracks the file descriptor
3. Intercepts `mmap` syscalls to detect where the library's text segment is mapped
4. Calculates function address: `base + nm_offset`
5. Begins recording only BBs within the target function range

**Example with llama.cpp:**
```bash
# Get offset from nm (one-time)
nm -D -S lib/libggml-cpu.so | grep gemv_q4_0_16x1_q8_0
# → 00000000000ac9a4 000000000000030a T ggml_gemv_q4_0_16x1_q8_0

# Run — no preliminary full-program run needed!
qemu-riscv64 -L llama-sysroot -E LD_LIBRARY_PATH=llama-lib -cpu max \
  -plugin libbbv.so,lib_name=libggml-cpu,func_offset=0xac9a4,func_size=0x30a,interval=1000,outfile=output/gemv \
  llama-bin/llama-completion -m model.gguf -p "Hello" -n 20
```

**Note:** The function must actually be called during execution. Use `nm` to
verify the offset matches the library version being loaded.

### Method 3: Direct Address Mode

For when you already know the runtime address (e.g., from a previous run):

```bash
setarch x86_64 -R \
  qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib \
  -plugin libbbv.so,func_addr=<decimal_addr>,func_size=<decimal_size>,interval=1000,outfile=output/gemv \
  ./binary
```

Requires ASLR disabled (`setarch x86_64 -R`) for consistent addresses.

## Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `func_name` | Symbol name (main program) | `ggml_gemv_q4_0_16x1_q8_0` |
| `lib_name` | Library name for syscall detection | `libggml-cpu` |
| `func_offset` | Static offset from nm (hex) | `0xac9a4` |
| `func_size` | Function size (hex or decimal) | `0x30a` or `778` |
| `func_addr` | Direct runtime address (decimal) | `140737327787502` |
| `interval` | BBV output interval | `10000` |
| `outfile` | Output file prefix | `output/result` |

## Output Format

### .bb file
SimPoint-compatible format:
```
T:<bb_index>:<count> ...
```

### .disas file
- Header: Function info (mode, address, size, range)
- Basic blocks: Address, instruction count, disassembly

## Detection Methods Summary

| Method | Use Case | Requires ASLR off | Two-pass |
|--------|----------|-------------------|----------|
| `func_name` | Main program symbols | No | No |
| `lib_name` + `func_offset` | Shared libraries | No | No |
| `func_addr` | Known runtime address | Yes | No |

## Limitations

- **Symbol-based detection** (`func_name`): Only works for main program symbols — `qemu_plugin_insn_symbol()` does not resolve shared library symbols
- **Syscall-based detection** (`lib_name`): The target function must actually execute during the run; if it doesn't, no BBV data is recorded
- **Direct address mode** (`func_addr`): Requires ASLR disabled for consistent addresses across runs
- **RISC-V only**: Syscall numbers (openat=56, mmap=222) are RISC-V specific
