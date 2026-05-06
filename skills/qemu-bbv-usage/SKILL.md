---
name: qemu-bbv-usage
description: |
  How to use QEMU (qemu-riscv64) with the custom BBV plugin for RISC-V program profiling
  within the RVFuse project. Use this skill whenever the user mentions running QEMU,
  qemu-riscv64, BBV profiling, RISC-V emulation, QEMU plugins, sysroot, running RISC-V binaries,
  QEMU CPU selection, vector extension programs, function-scoped recording, VLEN configuration,
  gemv profiling, library detection, or any RISC-V performance analysis — even if they don't
  explicitly ask for "QEMU help" or "BBV profiling."
---

# QEMU + BBV Plugin Usage Guide

This skill covers **runtime usage** of QEMU with the custom BBV (Basic Block Vector) plugin.
For plugin implementation details, see `references/implementation.md`.
For hotspot analysis best practices, see `references/analysis-guide.md`.

## Key Paths

| Item | Path |
|------|------|
| QEMU binary | `third_party/qemu/build/qemu-riscv64` |
| BBV plugin | `tools/bbv/libbbv.so` |
| Sysroot | `output/sysroot` or application-specific |
| BBV output | `outfile.<vcpu_index>.bb` + `outfile.disas` |

## BBV Plugin Features

- **Standard mode**: Profile entire program execution
- **Function-scoped mode**: NEW — Profile only specific target function
- **Syscall-based detection**: Automatically find shared library load addresses
- **Per-vCPU output**: Multi-threaded program support

## Running RISC-V Programs

```bash
qemu-riscv64 -L output/sysroot ./your_program [args...]
```

### CPU and VLEN Selection

**Critical for vector programs**: Match QEMU VLEN to compile-time `zvl*b` extension.

| Compile VLEN | QEMU Flag | Mismatch Result |
|--------------|-----------|-----------------|
| `zvl128b` (default) | `-cpu max` | OK |
| `zvl256b` | `-cpu max,vlen=256` | Illegal instruction or silent errors |
| `zvl512b` | `-cpu max,vlen=512` | Illegal instruction or silent errors |

```bash
# Enable V extension with specific VLEN
qemu-riscv64 -cpu max,vlen=256 -L output/sysroot ./vlen256_program

# Verify VLEN in program output
# Look for: RVV_VLEN = 32 (means 32 bytes = 256 bits)
```

### Debugging Flags

```bash
qemu-riscv64 -d strace -L output/sysroot ./program   # Syscall trace
qemu-riscv64 -d cpu -L output/sysroot ./program      # Register state
qemu-riscv64 -g 1234 -L output/sysroot ./program     # GDB remote debug
```

## BBV Plugin Usage

### Standard Recording (Full-Program)

```bash
qemu-riscv64 -L output/sysroot \
  -plugin tools/bbv/libbbv.so,interval=10000,outfile=output/profile \
  ./your_program [args...]
```

**Parameters**:
| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `outfile` | **Yes** | - | Output file prefix |
| `interval` | No | 100000000 | Instructions per sampling interval |

### Function-Scoped Recording (Recommended for Libraries)

Profile only a specific function within a shared library — much faster than full-program profiling.

#### Step 1: Get function offset from nm

```bash
nm -D -S output/llama.cpp/lib/libggml-cpu.so.0 | grep <function_name>
# Output: 00000000000aa7d8 000000000000030a T ggml_gemv_q4_0_16x1_q8_0
#         ^offset                           ^size
```

#### Step 2: Run with library detection

```bash
qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib -cpu max,vlen=256 \
  -plugin tools/bbv/libbbv.so,lib_name=libggml-cpu,func_offset=0xaa7d8,func_size=0x30a,interval=1000,outfile=output/gemv \
  ./llama-completion -m model.gguf -p "Hello" -n 20
```

**How it works**: Plugin intercepts `openat`/`mmap` syscalls to detect library loading,
then calculates function address and filters BB recording to that range only.

#### Function-Scoped Parameters

| Parameter | Format | Description |
|-----------|--------|-------------|
| `lib_name` | `libggml-cpu` | Library name (no `.so` suffix) — **requires** `func_offset` |
| `func_offset` | `0xaa7d8` | Static offset from nm (hex) |
| `func_size` | `0x30a` | Function size (hex or decimal) |
| `func_name` | `my_func` | Symbol name (main program only, not shared libs) |
| `func_addr` | `140737...` | Direct runtime address (decimal, requires ASLR off) |

#### Detection Methods Comparison

| Method | Use Case | Requires ASLR off | Setup |
|--------|----------|-------------------|-------|
| `lib_name` + `func_offset` | Shared libraries | No | nm offset |
| `func_name` | Main program symbols | No | Symbol name |
| `func_addr` | Known runtime address | Yes | Direct address |

### Output Files

**`.bb` file** — BBV execution counts (SimPoint format):
```
T:0:1000 1:500 2:250 ...
```

**`.disas` file** — Disassembly of recorded BBs:
```
# BBV Function-Scoped Mode (Syscall-Based)
# Library: libggml-cpu (base 0x7ffff62b9000)
# Function: offset 0xaa7d8, size 0x30a
# Range: 0x7ffff63637d8 - 0x7ffff6363ae2

BB 0 (vaddr: 0x7ffff63637d8, 2 insns):
  0x7ffff63637d8: addi a2,zero,16
  ...
```

### Interval Selection

| Interval | Granularity | Use Case |
|----------|-------------|----------|
| 1000-10000 | Fine | Function-scoped, short programs |
| 10000-100000 | Medium | Standard profiling |
| 100000+ | Coarse | Long-running programs |

## VLEN Configuration Example

For llama.cpp GEMV profiling (VLEN=256 kernel):

```bash
# 1. Verify binary has zvl256b
readelf -A output/llama.cpp/lib/libggml-cpu.so.0 | grep zvl256b
# Should show: zvl256b1p0

# 2. Get function offset
nm -D -S output/llama.cpp/lib/libggml-cpu.so.0 | grep gemv_q4_0_16x1_q8_0

# 3. Run with matching VLEN
qemu-riscv64 -L output/llama.cpp/sysroot \
  -E LD_LIBRARY_PATH=output/llama.cpp/lib \
  -cpu max,vlen=256 \
  -plugin tools/bbv/libbbv.so,lib_name=libggml-cpu,func_offset=0xaa7d8,func_size=0x30a,interval=1000,outfile=output/gemv \
  output/llama.cpp/bin/llama-completion \
  -m models/Qwen2.5-0.5B-Instruct-Q4_0.gguf -p "Hello" -n 10
```

See `applications/llama.cpp/README.md` section "GEMV Kernel Dispatch and VLEN Configuration"
for compile-time setup.

## BBV Output Analysis

```bash
python3 tools/analyze_bbv.py \
  --bbv output/profile.0.bb \
  --elf output/your_program \
  --sysroot output/sysroot \
  --top 20 \
  -o output/hotspot-report.txt
```

**Important**: Use the correct sysroot for your application:
- llama.cpp → `output/llama.cpp/sysroot`
- ONNX Runtime → `output/sysroot`

For detailed analysis best practices, see `references/analysis-guide.md`.

## Troubleshooting

### Quick Reference

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing libraries | "No such file" | Use correct `-L sysroot` |
| Missing outfile | "outfile unspecified" | Add `outfile=output/test` |
| Parameter format error | "option parsing failed" | No spaces, comma-separated |
| Missing func_offset | "func_offset required" | Pair `lib_name` with `func_offset` |
| Empty function output | 0 BBs | Verify function called, check nm offset |
| VLEN mismatch | Kernel not dispatched | Match QEMU vlen to compile zvl*b |
| Wrong sysroot | Hotspots in system libs | Use app-specific sysroot |
| PIE addresses | Main binary unmatched | Script auto-detects PIE |
| Stripped libs | `??` symbols | Use `nm -D` for dynamic symbols |

### Common Fixes

```bash
# Wrong: spaces in parameters
-plugin libbbv.so,lib_name = libggml-cpu

# Correct: no spaces
-plugin libbbv.so,lib_name=libggml-cpu,func_offset=0xaa7d8

# Verify binary VLEN requirement
readelf -A lib/libggml-cpu.so.0 | grep zvl

# Verify function exists and check offset
nm -D -S lib/libggml-cpu.so.0 | grep <function>
```

### Function Not Called

If function-scoped output is empty:
1. Verify function name in nm output (may have changed after rebuild)
2. Check program actually uses that code path (e.g., VLEN dispatch)
3. For llama.cpp GEMV: ensure `-DGGML_RV_ZVL256B=ON` was used in build

## Reference Files

| File | Content |
|------|---------|
| `references/implementation.md` | Plugin data structures, callbacks, syscall flow |
| `references/analysis-guide.md` | Hotspot analysis best practices |
| `tools/bbv/README.md` | Plugin build instructions |

## Building the Plugin

```bash
make -C tools/bbv/
./verify_bbv.sh  # Build + test
```

## Limitations

- `func_name` only works for main program symbols (not shared library symbols)
- `lib_name` requires target function to actually execute during the run
- `func_addr` requires ASLR disabled (`setarch x86_64 -R`)
- Syscall numbers (openat=56, mmap=222) are RISC-V specific