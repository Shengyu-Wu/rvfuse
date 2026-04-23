# QEMU Profiling vs Native Profiling: Function Path Divergence

## Problem Statement

When profiling llama.cpp Q4_0 inference, the functions executed under **QEMU user-mode emulation** differ significantly from those executed on **native RISC-V hardware**. This makes QEMU-based BBV profiling unreliable for capturing the actual hot functions observed in native `perf` data.

### Observed Divergence

| Function | Native perf (VLEN=256) | QEMU (VLEN=128) | QEMU (VLEN=512) |
|----------|----------------------|------------------|------------------|
| `ggml_gemv_q4_0_16x1_q8_0` | 27.82% (top-1) | Not called | Not called |
| `ggml_gemv_q8_0_16x1_q8_0` | 16.44% (top-2) | Not called | Not called |
| Any RVV gemv variant | — | Not called | Not called |

Under QEMU, **no RVV-optimized gemv function is dispatched at all** — the code falls through to a generic (scalar) path.

## Root Cause: VLEN-Dependent Dispatch

llama.cpp selects gemv/gemm implementations at runtime based on `__riscv_vlenb()`, which reports the CPU's vector length in bytes.

### Dispatch Logic (repack.cpp:4589-4598)

```cpp
if (ggml_cpu_has_riscv_v()) {
    #if defined __riscv_zvfh
    switch (__riscv_vlenb() * 8) {
        case 128:  { break; }           // TODO — no implementation
        case 256:  { if (cur->ne[1] % 16 == 0) { return &q4_0_16x1_q8_0; } break; }
        case 512:  { break; }           // TODO — no implementation
        case 1024: { break; }           // TODO — no implementation
        default:   { return nullptr; }
    }
    #endif
}
```

**The `16x1` RVV gemv is only dispatched when VLEN=256.** For VLEN=128 and VLEN=512, all cases fall through to `break` (no selection), and the function returns `nullptr`, causing a fallback to scalar generic code.

### Native Hardware

The perf data (`perf_q4_v.txt`) was collected on a **SpacemiT K1/X60** platform with **VLEN=256**, which matches the `case 256` branch — hence `ggml_gemv_q4_0_16x1_q8_0` was the top hot function.

## Compile Configuration

### Current Build Parameters

| Parameter | Value |
|-----------|-------|
| **Toolchain** | LLVM 22 (from `third_party/llvm-install`) |
| **Toolchain file** | `applications/llama.cpp/riscv64-linux-toolchain.cmake` |
| **C/C++ -march** | `rv64gcv_zfh_zba_zicbop` |
| **CMake flags** | `-DGGML_RVV=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZICBOP=ON -DGGML_RV_ZIHINTPAUSE=ON -DGGML_RV_ZVFH=ON` |
| **Binary attributes** | `rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicbop1p0_zicsr2p0_zifencei2p0_zihintpause2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zfh1p0_zfhmin1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvfh1p0_zvfhmin1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0` |
| **VLEN** | 128 (implied by `zvl128b`, no explicit `zvl256b`) |

### Key Observation

The current `-march=rv64gcv_zfh_zba_zicbop` does **not** include `zvl256b`. LLVM defaults to `zvl128b` for the V extension, so `__riscv_vlenb` is compiled as the constant `16` (VLEN=128). This means the `switch(__riscv_vlenb() * 8)` is optimized to `switch(128)`, which hits the `case 128: { break; }` — **dead code elimination removes the VLEN=256 dispatch entirely**.

Even if QEMU were set to VLEN=256, the compiled binary has already resolved `__riscv_vlenb()` to 128 at compile time, so the VLEN=256 code path is never reached.

## QEMU Configuration

| Parameter | Value |
|-----------|-------|
| **QEMU version** | 9.2.4 (from submodule `third_party/qemu`, tag `v9.2.4`) |
| **Default VLEN** | `-cpu max` → VLEN=128 (QEMU source: `cpu->cfg.vlenb = 128 >> 3`) |
| **Custom VLEN** | `-cpu max,vlen=256` or `-cpu max,vlen=512` — works for `--version` but `Illegal instruction` with full inference (possible extension mismatch) |
| **Run command** | `qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib -cpu max ./llama-completion -m model.gguf -p "Hello" -n 10` |

### VLEN Mismatch Matrix

| QEMU VLEN | Binary VLEN (compile-time) | `__riscv_vlenb` resolves to | Dispatch outcome |
|-----------|--------------------------|----------------------------|-----------------|
| 128 (`-cpu max`) | 128 (`zvl128b`) | 16 bytes = 128 bits | `case 128: break` — no RVV gemv |
| 256 (`-cpu max,vlen=256`) | 128 (`zvl128b`) | 16 bytes = 128 bits (compile-time!) | `case 128: break` — no RVV gemv |
| 512 (`-cpu max,vlen=512`) | 128 (`zvl128b`) | 16 bytes = 128 bits (compile-time!) | `case 128: break` — no RVV gemv |

**Changing QEMU's VLEN alone does not fix the problem** because the binary was compiled with `__riscv_vlenb` resolved at compile time.

## Native Profiling Platform

| Parameter | Value |
|-----------|-------|
| **Data file** | `applications/llama.cpp/benchmark/data/perf_q4_v.txt` |
| **Event** | `cpu-clock` |
| **Top functions** | `ggml_gemv_q4_0_16x1_q8_0` (27.82%), `ggml_gemv_q8_0_16x1_q8_0` (16.44%) |
| **Hardware** | RISC-V with VLEN=256 (SpacemiT K1/X60, per project context) |

## Solution Path

To make QEMU profiling match native hardware behavior, the **binary must be recompiled with `zvl256b`** so that `__riscv_vlenb` resolves to 32 (VLEN=256) at compile time, and QEMU must be run with matching VLEN=256.

### Required Changes

1. **Toolchain file** (`riscv64-linux-toolchain.cmake`):
   ```
   -march=rv64gcv_zfh_zba_zicbop
   → -march=rv64gcv_zfh_zba_zicbop_zvl256b
   ```

2. **Or, add `-D__riscv_v_fixed_vlen=256`** to CFLAGS (overrides `__riscv_vlenb` at compile time even without `zvl256b`)

3. **QEMU run command**:
   ```
   qemu-riscv64 -L sysroot -E LD_LIBRARY_PATH=lib -cpu max,vlen=256 ...
   ```

4. **Verify**: After rebuild, check `readelf -A libggml-cpu.so` shows `zvl256b` in the architecture string.

### Risk: VLEN=256 Binary on VLEN=128 Hardware

A binary compiled with `zvl256b` assumes VLEN >= 256 at runtime. Running it on VLEN=128 hardware (or QEMU without `vlen=256`) may produce incorrect results or crash. This is acceptable for our profiling use case where we control both the binary and QEMU configuration.

## Appendix: Full Dispatch Table for Q4_0

From `repack.cpp`, the Q4_0 gemv dispatch path in priority order:

1. **AVX2 / SVE**: `8x8` variant (x86)
2. **NEON + MATMUL_INT8**: `4x8` variant (ARM)
3. **NEON + DOTPROD**: `4x4` variant (ARM)
4. **RISC-V V + ZVFH**:
   - VLEN=128: `break` (no implementation)
   - VLEN=256: `16x1` variant ← **the one observed in native perf**
   - VLEN=512: `break` (no implementation)
   - VLEN=1024: `break` (no implementation)
5. **Fallback**: `nullptr` → scalar generic path

No RISC-V RVV path exists for VLEN=128 or VLEN=512 for Q4_0. The `16x1` variant is the **only** RVV gemv for Q4_0, and it requires VLEN=256.
