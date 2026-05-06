# eltwise-mul

RVV implementation of `MlasEltwiseMul<float>` — element-wise float multiplication using RISC-V Vector Extension.

## Status

✅ Implementation complete — straightforward vectorization of element-wise multiply.

## Files

| File | Purpose |
|------|---------|
| `rvv_eltwise_mul.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
template <>
void MlasEltwiseMul<float>(
    const float* left,
    const float* right,
    float* output,
    size_t N);
```

## Algorithm

1. Set VL dynamically via `vsetvl` for tail element handling
2. Load `left[i..i+vl-1]` via `vle32.v`
3. Load `right[i..i+vl-1]` via `vle32.v`
4. Multiply: `vfmul.vv`
5. Store result via `vse32.v`

Trivially vectorizable — each element independent.

## Instruction Reduction

For 8 elements (VLEN=256, VL=8):
- Scalar: 8 × (2 loads + 1 mul + 1 store) = 32 instructions
- RVV: 2 vle32.v + 1 vfmul.vv + 1 vse32.v + loop = ~5 instructions
- **~6x instruction reduction**

## Performance Impact

`MlasEltwiseMul` is 0.81% of YOLO inference CPU time directly, but it's also called from
`QuickGelu` (7.97% CPU time) after the logistic activation. Vectorizing this function
contributes to the overall QuickGelu acceleration.

## VLEN Requirement

- Works with any VLEN >= 128 (uses `vsetvl` for dynamic tail handling)

## Build & Test

```bash
# Build standalone test
riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
    --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
    -march=rv64gcv -mabi=lp64d \
    -D__riscv_v_intrinsic \
    test.cpp -o test -lm

# Run under QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=256 -L <sysroot> ./test
```

## Patch Integration

Modifies `onnxruntime/core/mlas/lib/eltwise.cpp`:
- Adds `#include "rvv_eltwise_mul.inl"`
- Updates `MlasEltwiseMul<float>` to dispatch to RVV path on RISC-V targets

## References

- Scalar implementation: `onnxruntime/core/mlas/lib/eltwise.cpp`
- QuickGelu (consumer): `onnxruntime/contrib_ops/cpu/activations.h`
- Perf analysis: `../../yolo/data/perf/yolo11n_bananapi_k1_rv64gcv_scalar_20260424_analysis.md`
