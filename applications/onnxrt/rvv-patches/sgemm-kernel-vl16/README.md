# sgemm-kernel-vl16

RVV implementation of `MlasSgemmKernel` (VL=16) + `MlasSgemmPackB16` — single-precision GEMM kernel targeting 512-bit vector registers (VLEN=512, SEW=32, VL=16).

## Status

✅ Implementation complete — 4x wider than the VL=4 kernel, requires 16-column B packing.

## Files

| File | Purpose |
|------|---------|
| `rvv_sgemm_kernel_vl16.inl` | RVV VL=16 GEMM kernel (single source of truth) |
| `rvv_sgemm_pack_b16.inl` | B matrix 16-column packing (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signatures

```cpp
// VL=16 GEMM kernel
size_t MlasSgemmKernelRvv512(
    const float* A, const float* B, float* C,
    size_t CountK, size_t CountM, size_t CountN,
    size_t lda, size_t ldc, float alpha, bool ZeroMode);

// B16 packing
void MlasSgemmPackB16(
    float* D, const float* B, size_t ldb,
    size_t CountX, size_t CountY);
```

## Algorithm

### GEMM Kernel (VL=16)
1. Set VL=16 (512-bit VLEN, SEW=32, LMUL=1)
2. For each 16-column output block:
   - Zero 16-element accumulators for 1-2 rows
   - Inner K loop unrolled by 2:
     a. Load A elements (scalar, broadcast via `vfmacc.vf`)
     b. Load B[0..15] vector (K element 0) → `vfmacc.vf`
     c. Load B[16..31] vector (K element 1) → `vfmacc.vf`
   - Handle odd K element
   - Multiply by alpha
   - Store full 16-element block or partial remainder

### B16 Packing
- Pack B matrix columns into 16-element contiguous blocks
- Layout: `D[col0..col15]` per K row, 16 floats per row
- Zero-padding for partial blocks (< 16 columns)
- Total packed size: CountK × 16 floats × ceil(CountN/16) blocks

## Performance Comparison

| Metric | VL=4 Kernel | VL=16 Kernel | Ratio |
|--------|-------------|--------------|-------|
| Output cols/iter | 4 | 16 | 4× |
| K-loop ops (unrolled×2) | 10 | 8 | 0.8× |
| Efficiency (cols/op) | 0.4 | 2.0 | 5× |

## VLEN Requirement

- **VLEN >= 512**: Uses VL=16 RVV intrinsics (requires 512-bit / SEW=32 = 16 elements)
- **VLEN < 512**: Falls back to scalar kernel (MLAS reference implementation)

## Build & Test

```bash
# Build standalone test (VLEN=512)
riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv_zvl512b -mabi=lp64d \
    -D__riscv_v -D__riscv_v_intrinsic -DVLEN_512 -DMLAS_RVV_VLEN_512 \
    -DMLAS_TARGET_RISCV \
    test.cpp -o test -lm

# Run under QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=512 -L <sysroot> ./test
```

## Patch Integration

Builds on `sgemm-kernel` (VL=4). Adds:
- VL=16 kernel (`riscv64/SgemmKernelRvv512.cpp`)
- B16 packing (`riscv64/SgemmPackB16.cpp`)
- VLEN-based dispatch: `GemmFloatKernel` uses VL=16 kernel when `MLAS_RVV_VLEN_512` is defined
- B packing dispatch in `sgemm.cpp`: uses 16-column packing for VL=16

## References

- Perf analysis: `applications/onnxrt/yolo/data/perf/` — SGEMM is 77% of YOLO inference CPU time
