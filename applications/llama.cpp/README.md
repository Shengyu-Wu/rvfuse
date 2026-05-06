# rv64gcv-llama.cpp

Cross-compile llama.cpp for RISC-V 64-bit with vector extension (RVV 1.0).

## Build Status

**Successfully compiled** (2026-04-14)

| Component | Status |
|-----------|--------|
| llama-cli | ✓ RISC-V ELF (rv64gcv) |
| llama-server | ✓ RISC-V ELF (rv64gcv) |
| libllama.so | ✓ Shared library |
| libggml-cpu.so | ✓ RVV-optimized backend |
| Total binaries | 43 executables |

## Overview

This directory contains build scripts for cross-compiling llama.cpp to RISC-V rv64gcv target using LLVM 22. The build enables:

- **RVV**: RISC-V Vector extension (v1.0) for auto-vectorization
- **ZFH**: Half-precision float support
- **ZICBOP**: Cache block operations for prefetch
- **ZIHINTPAUSE**: Pause hint for spin loop optimization
- **ZVFH**: Vector half-precision float (auto-enabled by llama.cpp cmake)

Target architecture: `rv64gcv_zfh_zvfh_zicbop_zihintpause` (lp64d ABI)

## Directory Structure

```
applications/llama.cpp/
├── README.md
├── build.sh                 # Cross-compile orchestrator
├── models/                  # GGUF model files
├── output/                  # Build artifacts
├── vendor/                  # llama.cpp source (git submodule)
├── qwen                     # Qwen inference wrapper
├── riscv64-linux-toolchain.cmake
│
├── rvv-patches/             # RVV implementations (inl + patch + test)
│   ├── gemm-q4_K-8x4-q8_K/   # Q4_K × Q8_K GEMM (4x4 tile)
│   │   ├── rvv_gemm_q4_K_8x4.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   ├── gemv-q4_K-8x8-q8_K/  # Q4_K × Q8_K GEMV (8x8 tile)
│   │   ├── rvv_gemv_q4_K_8x8_q8_K.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   ├── quantize-q8_0-4x4/   # FP32 → Q8_0 quantize (4x4 interleaved)
│   │   ├── rvv_quantize_q8_0_4x4.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   ├── vec-dot-q5_0-q8_0/  # Q5_0 × Q8_0 vector dot product
│   │   ├── rvv_vec_dot_q5_0_q8_0.inl
│   │   ├── patch.diff
│   │   ├── test.cpp
│   │   └── README.md
│   │
│   └── _template/           # Template for new RVV implementations
│       ├── rvv_<name>.inl.template
│       ├── patch.diff.template
│       ├── test.cpp.template
│       └── README.md.template
```

Each RVV implementation follows the **single source of truth** principle:
- `.inl` file contains the RVV implementation code
- `patch.diff` applies changes to llama.cpp (includes the .inl file)
- `test.cpp` validates correctness against scalar reference

## Prerequisites

- LLVM 22 installation at `third_party/llvm-install/`
- cmake >= 3.14, ninja, git
- Docker (for sysroot extraction, optional if using shared sysroot)

## Build

```bash
# Standard build (uses shared sysroot from rv64gcv-onnxrt if available)
./build.sh

# Standalone mode (extract own sysroot via Docker)
./build.sh --standalone

# Force rebuild everything
./build.sh --force -j 8

# Incremental build (skip source cloning)
./build.sh --skip-source

# Quick rebuild (skip source and sysroot)
./build.sh --skip-source --skip-sysroot
```

## Output

Artifacts are placed in `output/llama.cpp/` (~596M total):

### Executables (`bin/`)
| Tool | Size | Description |
|------|------|-------------|
| llama-cli | 4.1M | Command-line inference tool |
| llama-server | 3.0M | OpenAI-compatible HTTP server |
| llama-bench | 1.2M | Performance benchmarking |
| llama-quantize | 1.5M | Model quantization tool |
| llama-perplexity | 2.9M | Model evaluation |
| llama-embedding | 2.9M | Embedding generation |
| llama-batched | 2.9M | Batched inference |
| *(35 more)* | - | Various tools and examples |

### Libraries (`lib/`)
| Library | Size | Description |
|---------|------|-------------|
| libllama.so | 2.4M | Core llama library |
| libggml.so | 34K | GGML base library |
| libggml-cpu.so | 734K | CPU backend (RVV-optimized) |
| libggml-base.so | 667K | GGML base functions |
| libmtmd.so | 675K | Multimodal support |

### Sysroot (`sysroot/`)
RISC-V Linux sysroot (~220M) with:
- libc, libm, libdl, librt, libpthread
- libstdc++, libgcc_s
- Dynamic linker: `ld-linux-riscv64-lp64d.so.1`

## Quick Start

The `qwen` wrapper script provides easy access to Qwen inference via QEMU:

```bash
# Simple inference (auto-downloads model if missing)
./qwen -p "Hello, world!"

# Custom options
./qwen -p "What is 2+2?" -n 10 --temp 0.5

# Interactive chat mode
./qwen -i

# Multi-threaded (note: QEMU is slow, threads help little)
./qwen -p "Tell me a story" -t 4 -n 64
```

Options:
| Flag | Description | Default |
|------|-------------|---------|
| `-p, --prompt` | Prompt text | Required |
| `-n, --tokens` | Tokens to generate | 32 |
| `-t, --threads` | Number of threads | 1 |
| `--temp` | Temperature | 0.7 |
| `-i, --interactive` | Chat mode | false |
| `-m, --model` | Model filename | Qwen2.5-0.5B-Instruct-Q4_0.gguf |
| `-h, --help` | Show help | - |

**Note**: QEMU emulation is ~50-100x slower than native RISC-V hardware. Expect 0.2-0.3 tokens/second.

## Usage with QEMU (Advanced)

Direct llama.cpp execution without the wrapper:

```bash
SYSROOT=output/llama.cpp/sysroot
BIN=output/llama.cpp/bin

# Run inference (requires GGUF model file)
qemu-riscv64 -L $SYSROOT -cpu max $BIN/llama-cli \
    -m models/Qwen2.5-0.5B-Instruct-Q4_0.gguf \
    -p "Hello, world!" \
    -t 4

# Run HTTP server
qemu-riscv64 -L $SYSROOT $BIN/llama-server \
    -m models/qwen-0.5b-q4_0.gguf \
    --port 8080

# Benchmark performance
qemu-riscv64 -L $SYSROOT $BIN/llama-bench \
    -m models/qwen-0.5b-q4_0.gguf
```

## Model Preparation

llama.cpp requires GGUF format models. Download pre-converted models:

```bash
# From HuggingFace (Qwen2.5 0.5B Q4_0)
mkdir -p models
wget -O models/qwen-0.5b-q4_0.gguf \
    https://huggingface.co/ggml-org/Qwen2.5-0.5B-GGUF/resolve/main/qwen2.5-0.5b-q4_0.gguf
```

Or convert from HuggingFace format:

```bash
python3 output/llama.cpp/bin/convert_hf_to_gguf.py \
    --outfile models/output.gguf \
    --outtype q4_0 \
    <hf-model-dir>
```

## Version

| Component | Version |
|-----------|---------|
| llama.cpp | **`b8783`** ( pinned, commit `e21cdc11`) |
| LLVM | 22.1.3 |
| Target | `rv64gcv_zfh_zvfh_zicbop_zihintpause` |
| ABI | lp64d |

> **Do not upgrade `vendor/llama.cpp`** without verifying that the RVV GEMV implementations
> in `ggml/src/ggml-cpu/arch/riscv/repack.cpp` still match the expected function signatures.
> The `ggml_gemv_q4_K_8x8_q8_K` analysis and patches in this branch are based on `b8783`.

## BBV Profiling Hotspot Analysis (Q4_0 Model)

Profiling conducted on Qwen2.5-0.5B-Instruct **Q4_0** quantized model (2026-04-15).

Test parameters: `llama-bench -p 32 -n 0 -r 1 -t 1`

### Note on Test Limitations

This short benchmark (32 prompt tokens, 0 generation) shows **83%** execution time in model initialization (hashtable rehash, vocabulary loading). The inference compute section below focuses on the remaining **17%** representing actual computation.

For accurate inference profiling, use longer prompts: `-p 512 -n 32`.

### Inference Compute Hotspots (Core Functions Only)

Core inference functions (quantize + GEMV) represent **5.5%** of total execution in this test.

| Category | Share of Core |
|----------|---------------|
| Quantize (activation → Q8_0/Q8_K) | **60.7%** |
| GEMV (matrix-vector multiply) | **39.3%** |

#### Top Inference Functions

| Function | Library | % of Core | Description |
|----------|---------|-----------|-------------|
| `ggml_quantize_mat_q8_0_4x4` | libggml-cpu.so | **49.8%** | Quantize FP32 activations → Q8_0 (4x4 interleaved) |
| `ggml_gemv_q4_K_8x8_q8_K` | libggml-cpu.so | **32.1%** | Q4_K weight × Q8_K activation GEMV |
| `ggml_gemv_iq4_nl_8x8_q8_0` | libggml-cpu.so | 5.3% | IQ4_NL weight × Q8_0 activation GEMV |
| `ggml_quantize_mat_q8_K_4x1` | libggml-cpu.so | 3.9% | Quantize activations → Q8_K |
| `dequantize_row_iq2_xs` | libggml-base.so | 3.1% | Dequantize IQ2_XS weights → FP32 |
| `ggml_quantize_mat_q8_K_4x4` | libggml-cpu.so | 2.0% | Quantize activations → Q8_K (4x4) |
| `dequantize_row_iq4_nl` | libggml-base.so | 1.9% | Dequantize IQ4_NL weights → FP32 |
| `ggml_gemv_iq4_nl_4x4_q8_0` | libggml-cpu.so | 1.3% | IQ4_NL × Q8_0 GEMV (smaller block) |

### Quantization Flow (Q4_0 Model)

```
┌────────────────────────────────────────────────────────────────────┐
│                    Inference Compute Pipeline                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Weight Matrix         Activation Vector (FP32)                   │
│   Format: Q4_0/IQ4_NL            ↓                                 │
│         ↓              ggml_quantize_mat_q8_0_4x4                   │
│   Runtime repack       (FP32 → Q8_0, ~50% of compute)              │
│   to Q4_K (optional)              ↓                                 │
│         ↓                                                         │
│   ggml_gemv_q4_K_8x8_q8_K  or  ggml_gemv_iq4_nl_8x8_q8_0           │
│   (~32% of compute)                                               │
│         ↓                                                         │
│                    FP32 Output                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Key Observations

1. **Quantize dominates GEMV** (60.7% vs 39.3%)
   - Activation quantization overhead is significant in short benchmarks
   - With longer sequences, GEMV ratio increases (attention is O(L²))

2. **Q4_K GEMV appears with Q4_0 model**
   - llama.cpp repacks Q4_0 weights → Q4_K at runtime for better vectorization
   - Q4_K has 8x8 block structure optimized for SIMD-style GEMV

3. **Multiple quantization formats**
   - Q8_0: 32-element blocks, used for IQ4_NL weights
   - Q8_K: 256-element blocks, used for Q4_K weights
   - Interleaving (4x4, 8x8) improves cache efficiency

### Expected Behavior with Longer Inference

| Prompt Length | Quantize | GEMV | Other Compute |
|---------------|----------|------|---------------|
| 32 tokens (current) | 60% | 40% | negligible |
| 512 tokens | ~25% | **~65%** | ~10% |
| 1024+ tokens | ~15% | **~80%** | ~5% |

### Q8_0 Model Inference Phase Analysis

Profiling conducted on Qwen2.5-0.5B-Instruct **Q8_0** quantized model (2026-04-15):
- Model size: 675MB (vs 428MB for Q4_0)
- Inference: 20 tokens generated via QEMU BBV profiling
- **Filtered**: Excluded initialization phase (`backend_load`, `numa_init`, `quantize_iq2_s`)

#### Inference Phase Hotspots Distribution

| Category | Function | Library | Execution % |
|----------|----------|---------|-------------|
| Batch Management | `llama_batch_allocr::split_equal` | libllama.so | **14.59%** |
| GEMM (Q4_K) | `ggml_gemm_q4_K_8x4_q8_K` | libggml-cpu.so | **7.07%** |
| GEMV (MXFP4) | `ggml_gemv_mxfp4_4x4_q8_0` | libggml-cpu.so | **1.11%** |
| Crypto | (checksum operations) | libcrypto.so | **5.73%** |

#### Library Distribution (Inference Phase)

| Library | Execution % |
|---------|-------------|
| libllama.so | **23.32%** |
| libggml-cpu.so | **16.96%** |
| libcrypto.so | 13.94% |
| libggml-base.so | 6.65% |

#### Category Distribution (Inference Phase)

| Category | Execution % |
|----------|-------------|
| Batch Management | **20.09%** |
| GEMV/GEMM (Matrix ops) | **16.22%** |
| Crypto (checksum) | 13.94% |
| Other | 15.41% |

#### GEMV/GEMM Function Breakdown

| Function | Purpose | % of Matrix Ops |
|----------|---------|-----------------|
| `ggml_gemm_q4_K_8x4_q8_K` | Q4_K weights × Q8_K activation | **52.9%** |
| `ggml_gemv_mxfp4_4x4_q8_0` | MXFP4 weights × Q8_0 activation | 13.9% |
| `ggml_gemv_q2_K_16x1_q8_K_generic` | Q2_K weights × Q8_K activation | 10.3% |
| `ggml_gemm_q4_0_8x8_q8_0_generic` | Q4_0 weights × Q8_0 activation | 9.8% |
| `ggml_gemm_q4_0_4x8_q8_0` | Q4_0 weights × Q8_0 activation | 5.9% |
| `ggml_gemv_q8_0_16x1_q8_0_generic` | Q8_0 weights × Q8_0 activation | 4.4% |

#### Key Findings: Q4 vs Q8 Comparison

| Metric | Q4_0 Model | Q8_0 Model |
|--------|------------|------------|
| Batch Management % | 77.75% | 20.09% |
| GEMV/GEMM % | 3.07% | 16.22% |
| Quantization % | 4.59% | 1.53% |
| Backend Load % | 1.62% | 14.79% (filtered) |

**Analysis**:

1. **Higher compute ratio in Q8**: GEMV/GEMM accounts for 16.22% vs 3.07% in Q4
   - Q8 weights are larger, requiring more matrix operations
   - Batch management overhead is relatively lower

2. **Repacking still dominant**: Even for Q8 model, `ggml_gemm_q4_K_8x4_q8_K` dominates (52.9%)
   - llama.cpp repacks weights to Q4_K format for efficient computation
   - This applies to both Q4 and Q8 quantized models

3. **Common activation format**: All GEMV/GEMM functions use Q8 activation (`q8_0` or `q8_K` suffix)
   - Q4 and Q8 models share the same activation quantization path
   - Weight format determines which kernel is selected

4. **Crypto overhead**: libcrypto.so accounts for 13.94% (checksum validation)
   - More prominent in Q8 due to larger model file validation

## Verifying RVV Kernel Execution

The standalone correctness tests (`test.cpp`) validate numerical accuracy against the scalar reference, but they do not prove the kernel is used in actual inference. This section describes how to confirm a patched kernel is called at runtime, and pitfalls to avoid.

### Why This Is Non-Trivial

llama.cpp loads `libggml-cpu.so` dynamically at runtime via `dlopen`. The build system has two artifact directories with separate copies of this library:

| Directory | Contents | Used by |
|-----------|----------|---------|
| `output/llama.cpp/.build/bin/` | Build output (raw) | `ninja` link step only |
| `output/llama.cpp/lib/` | Installed output | `qemu-riscv64` via `LD_LIBRARY_PATH` |

**Critical**: `ninja` alone does NOT update the installed `lib/` directory. Running the binary after `ninja` will load the **old** library without your changes. You must either:
- Run `build.sh` (which includes the install step), or
- Manually copy the rebuilt `.so` to `output/llama.cpp/lib/`

Additionally, the compiler inlines `static` functions (like `_rvv` suffix functions) and may eliminate trace code if not written carefully.

### Method 1: One-Time Trace Marker (Recommended)

Add a `volatile` guarded `write()` call inside the RVV function in the `.inl` file. `volatile` prevents the compiler from optimizing away the check; `write()` bypasses stdio buffering.

```c
#include <unistd.h>  // for write(), STDERR_FILENO

static void ggml_vec_dot_q5_0_q8_0_rvv(...) {
    // ... existing code ...

    // One-time trace: prints on first call, then silences
    {
        static volatile int _called = 0;
        if (_called == 0) {
            _called = 1;
            char buf[128];
            int len = snprintf(buf, sizeof(buf),
                "[RVV-TRACE] %s: n=%d, nb=%d, vlenb=%zu (VLEN=%zu)\n",
                __FUNCTION__, n, nb, vlenb, vlenb * 8);
            write(STDERR_FILENO, buf, len);
        }
    }

    // ... rest of function ...
}
```

**Why `volatile`**: Without `volatile`, LLVM 22 at `-O2` evaluates `static int _called = 0` as a compile-time constant and removes the entire branch, even though the variable would change at runtime. The trace silently disappears.

**Why `write()` not `fprintf()`**: `fprintf(stderr, ...)` may be buffered and not flushed before QEMU exits, especially when piping output through `grep` or `tee`.

**Build and run**:

```bash
# 1. Copy .inl to vendor source
cp rvv-patches/vec-dot-q5_0-q8_0/rvv_vec_dot_q5_0_q8_0.inl \
   vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/

# 2. Force recompile (inl changes don't update .o timestamps)
touch vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c

# 3. Rebuild
cd output/llama.cpp/.build && ninja -j4

# 4. IMPORTANT: copy rebuilt .so to installed location
cp bin/libggml-cpu.so.0.9.11 ../../lib/

# 5. Run with Q5_0-containing model (Q4_K_M uses mixed precision with Q5_0 tensors)
qemu-riscv64 -L output/llama.cpp/sysroot \
    -E LD_LIBRARY_PATH=output/llama.cpp/lib \
    output/llama.cpp/bin/llama-completion \
    -m models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
    -p "Hi" -n 4 -t 1 2>&1 | grep "RVV-TRACE"
```

Expected output:

```
[RVV-TRACE] ggml_vec_dot_q5_0_q8_0_rvv: n=896, nb=28, vlenb=16 (VLEN=128)
```

- `n=896`: Qwen2.5-0.5B embedding dimension (vector length)
- `nb=28`: 896 / 32 = 28 Q5_0 blocks per call
- `vlenb=16`: QEMU emulates VLEN=128 bit (16 bytes)
- The trace prints only once; the function is actually called thousands of times per inference

**Choosing the right model**: The kernel is only called for tensors quantized as Q5_0. Not all Q4 models contain Q5_0 tensors. `Q4_K_M` is a mixed-precision format where FFN layers use Q5_0 (~55% of compute, see `temp/perf_q4_hotspot_analysis.md`). A pure `Q4_0` model will NOT trigger `vec_dot_q5_0_q8_0` at all.

### Method 2: Disassembly Verification

If you cannot modify the source, verify the binary directly:

```bash
# Check that the function symbol exists and points to RVV instructions
llvm-objdump -d --start-address=<addr> --stop-address=<addr>+0x50 \
    output/llama.cpp/lib/libggml-cpu.so

# Or find the address first
nm -D output/llama.cpp/lib/libggml-cpu.so | grep "vec_dot_q5_0_q8_0$"
```

Expected: RVV instructions (`vle8.v`, `vlm.v`, `vwmul.vv`, `vwredsum.vs`) should appear in the disassembly, confirming the compiler chose the `#if defined(__riscv_v)` path.

### Method 3: Symbol Table Verification

Confirm the RVV function exists in the loaded library:

```bash
nm -D output/llama.cpp/lib/libggml-cpu.so | grep "vec_dot_q5_0_q8_0"
```

Expected output:

```
00000000000a77b6 T ggml_vec_dot_q5_0_q8_0           # arch-specific (calls _rvv)
0000000000066162 T ggml_vec_dot_q5_0_q8_0_generic   # scalar fallback
```

The `_generic` version is the fallback. If only `_generic` exists (no arch-specific symbol), the patch was not applied or the arch source was not compiled.

### Checklist for Adding New RVV Kernels

| Step | Command / Action | Pitfall |
|------|-----------------|---------|
| 1. Copy `.inl` | `cp rvv-patches/<name>/<name>.inl vendor/.../arch/riscv/` | File may already exist from previous build |
| 2. Apply patch | `git apply rvv-patches/<name>/patch.diff` | Patch format must match exact line counts |
| 3. Touch source | `touch vendor/.../arch/riscv/quants.c` (or repack.cpp) | `.inl` changes alone don't trigger recompile |
| 4. Rebuild | `cd output/llama.cpp/.build && ninja -j4` | `ninja` only updates `.build/bin/` |
| 5. Install `.so` | `cp bin/libggml-cpu.so.* ../../lib/` | **Most commonly missed step** |
| 6. Verify string | `strings output/llama.cpp/lib/libggml-cpu.so \| grep "RVV-TRACE"` | Confirms the new code is in the loaded library |
| 7. Run inference | Use a model that contains the target quant format | Wrong model = kernel never called, no trace output |
| 8. Remove trace | Restore clean `.inl` before committing | Don't leave debug code in production |

## GEMV Kernel Dispatch and VLEN Configuration

llama.cpp selects GEMV/GEMM kernels at runtime based on `__riscv_vlenb()`, which returns
the CPU's vector length in bytes. This creates a critical dependency between compile-time
march flags and runtime kernel selection.

### The VLEN Dispatch Mechanism

The dispatch logic in `vendor/llama.cpp/ggml/src/ggml-cpu/arch/riscv/repack.cpp` (around line 4589):

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

**Key insight**: `__riscv_vlenb()` is resolved at **compile time** when `zvl*b` extension is specified.
Without `zvl256b`, LLVM defaults to `zvl128b`, causing `__riscv_vlenb()` to resolve to `16` (VLEN=128).
The switch hits `case 128: break` — the VLEN=256 kernel path is **dead-code eliminated**.

### Required Build Configuration for VLEN=256

To ensure the `16x1` GEMV kernel (VLEN=256 variant) is compiled and dispatched:

#### 1. Toolchain file modification

**File**: `riscv64-linux-toolchain.cmake` (lines 23-24)

```cmake
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv_zfh_zba_zicbop_zvl256b")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv_zfh_zba_zicbop_zvl256b")
```

#### 2. CMake VLEN parameter (CRITICAL)

**Add to build.sh cmake configuration**:

```bash
cmake ... -DGGML_RV_ZVL256B=ON ...
```

This cmake option is processed by `rvv-patches/cmake-vlen-config/patch.diff`, which
adds VLEN configuration support to llama.cpp's CMake. The patch is automatically
applied by `build.sh`.

**Why this is needed**: llama.cpp's CMake constructs its own `MARCH_STR` that
overrides the toolchain file's march flags. Without this cmake option, LLVM defaults
to `zvl128b` (VLEN=128) and the VLEN=256 GEMV kernel path is dead-code eliminated.

#### 3. Verify the build

```bash
# After rebuild, check ELF attributes for zvl256b
readelf -A output/llama.cpp/lib/libggml-cpu.so.0 | grep zvl256b

# Expected output includes: zvl256b1p0
```

### Required QEMU Configuration

Even with `zvl256b` compiled in, QEMU must be configured to match:

```bash
# WRONG: Default QEMU VLEN=128 → Illegal instruction or silent failure
qemu-riscv64 -L sysroot -cpu max ./llama-cli ...

# CORRECT: Match VLEN=256
qemu-riscv64 -L sysroot -cpu max,vlen=256 ./llama-cli ...
```

Verification in program output:

```
system_info: n_threads = 11 ... | RISCV_V = 1 | RVV_VLEN = 32 | ...
```

`RVV_VLEN = 32` means 32 bytes = 256 bits. If it shows `RVV_VLEN = 16`, the kernel
will NOT be dispatched (VLEN mismatch).

### Adding New GEMV/GEMM Kernels for Different VLEN

When implementing new RVV GEMV/GEMM kernels, you must:

1. **Add kernel implementation**: Create `.inl` file in `vendor/.../arch/riscv/`
2. **Register in dispatch table**: Modify `repack.cpp` switch statement:

```cpp
switch (__riscv_vlenb() * 8) {
    case 128:  { return &my_new_128bit_kernel; break; }  // Add your kernel
    case 256:  { return &q4_0_16x1_q8_0; break; }         // Existing
    case 512:  { return &my_new_512bit_kernel; break; }   // Add your kernel
    ...
}
```

3. **Declare function pointer**: Add to the appropriate function pointer table in `repack.cpp`
4. **Enable VLEN cmake option**: Add appropriate cmake parameter to `build.sh`:

```bash
# For VLEN=256 kernels
-DGGML_RV_ZVL256B=ON

# For VLEN=512 kernels
-DGGML_RV_ZVL512B=ON

# For VLEN=1024 kernels
-DGGML_RV_ZVL1024B=ON
```

The cmake-vlen-config patch (in `rvv-patches/cmake-vlen-config/`) handles the CMakeLists.txt
modification automatically.

### VLEN Configuration Matrix

| Target VLEN | CMake march suffix | QEMU flag | ELF attribute | Dispatch case |
|-------------|-------------------|-----------|---------------|---------------|
| 128 | (default, no suffix) | `-cpu max` | `zvl128b` | `case 128` |
| 256 | `_zvl256b` | `-cpu max,vlen=256` | `zvl256b` | `case 256` |
| 512 | `_zvl512b` | `-cpu max,vlen=512` | `zvl512b` | `case 512` |
| 1024 | `_zvl1024b` | `-cpu max,vlen=1024` | `zvl1024b` | `case 1024` |

**Warning**: A binary compiled with `zvl256b` requires VLEN ≥ 256 at runtime.
Running on VLEN=128 hardware (or QEMU without `vlen=256`) may produce:
- `Illegal instruction` (vector ops exceed hardware VLEN)
- Silent incorrect results (partial vector initialization)

### BBV Profiling with VLEN=256 GEMV

To profile the `ggml_gemv_q4_0_16x1_q8_0` function with BBV plugin:

```bash
# 1. Get function offset from nm (after zvl256b rebuild)
nm -D -S output/llama.cpp/lib/libggml-cpu.so.0 | grep gemv_q4_0_16x1_q8_0
# Example output: 00000000000aa7d8 000000000000030a T ggml_gemv_q4_0_16x1_q8_0

# 2. Run QEMU with BBV plugin targeting the function
qemu-riscv64 -L sysroot \
  -E LD_LIBRARY_PATH=output/llama.cpp/lib \
  -cpu max,vlen=256 \
  -plugin tools/bbv/libbbv.so,lib_name=libggml-cpu,func_offset=0xaa7d8,func_size=0x30a,interval=1000,outfile=output/gemv-profile \
  output/llama.cpp/bin/llama-completion \
  -m models/Qwen2.5-0.5B-Instruct-Q4_0.gguf \
  -p "Hello" -n 10
```

Expected BBV output verification:

```bash
# Check disas header shows correct function range
head -5 output/gemv-profile.disas
# Should show: Function: offset 0xaa7d8, size 0x30a, Range: <base> - <end>

# Check BB data is non-empty
wc -l output/gemv-profile.0.bb
# Should show thousands of lines (function was called)
```

## References

- [llama.cpp RISC-V documentation](https://github.com/ggerganov/llama.cpp/blob/master/docs/build-riscv64-spacemit.md)
- [llama.cpp build guide](https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md)
- [RVV 1.0 specification](https://github.com/riscv/riscv-v-spec)