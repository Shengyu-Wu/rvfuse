# CMake VLEN Configuration Patch

## Purpose

Add cmake options `GGML_RV_ZVL256B`, `GGML_RV_ZVL512B`, `GGML_RV_ZVL1024B` to control
vector length (VLEN) during compilation. This is required for dispatching the correct
GEMV/GEMM kernels at runtime.

## Background

llama.cpp selects kernels at runtime based on `__riscv_vlenb()`, which is resolved at
compile time when `zvl*b` extension is specified. Without explicit VLEN configuration,
LLVM defaults to `zvl128b` (VLEN=128), causing VLEN=256 kernels to be dead-code eliminated.

See `applications/llama.cpp/README.md` section "GEMV Kernel Dispatch and VLEN Configuration"
for details.

## Usage

Enable in `build.sh`:

```bash
cmake ... -DGGML_RV_ZVL256B=ON ...
```

Or for larger VLEN:

```bash
cmake ... -DGGML_RV_ZVL512B=ON ...
cmake ... -DGGML_RV_ZVL1024B=ON ...
```

## Files Modified

- `ggml/src/ggml-cpu/CMakeLists.txt`: Add VLEN cmake options after `GGML_RV_ZIHINTPAUSE`

## Testing

After build, verify ELF attributes:

```bash
readelf -A output/llama.cpp/lib/libggml-cpu.so.0 | grep zvl
# Expected: zvl256b1p0 (or zvl512b1p0, etc.)
```

## Notes

- Only one VLEN option should be enabled at a time
- VLEN must match target hardware (or QEMU configuration)
- Running a VLEN=256 binary on VLEN=128 hardware may produce `Illegal instruction`