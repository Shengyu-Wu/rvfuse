# compute-logistic

RVV implementation of `MlasLogisticKernel` — sigmoid (logistic) activation function using RISC-V Vector Extension.

## Status

✅ Implementation complete — vectorized polynomial approximation of the sigmoid function.

## Files

| File | Purpose |
|------|---------|
| `rvv_compute_logistic.inl` | RVV implementation (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime MLAS |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
void MlasLogisticKernel(const float* Input, float* Output, size_t N);
void MlasComputeLogistic(const float* Input, float* Output, size_t N);
```

## Algorithm

Polynomial approximation of the sigmoid function (same as Eigen):

```
sigmoid(x) = 1 / (1 + exp(-x))
```

1. Clamp input to [-18, 18] (NaN-safe bounds)
2. Compute p(x) = x * (α₁ + x² * (α₃ + x² * (α₅ + x² * (α₇ + x² * α₉))))
3. Compute q(x) = β₀ + x² * (β₂ + x² * (β₄ + x² * (β₆ + x² * (β₈ + x² * β₁₀))))
4. Result = clamp(p(x)/q(x) + 0.5, 0, 1)

### RVV Vectorization
- Uses `vsetvl` for tail element handling (any N)
- `vfmacc.vf` for FMA polynomial evaluation (Horner's method)
- `vfdiv.vv` for final p/q division
- `vfmax.vf` / `vfmin.vf` for clamping

## Instruction Reduction

For 8 elements (VLEN=256, VL=8):
- Scalar: 8 × (1 clamp + 1 x² + 10 FMA + 1 div + 1 add + 1 clamp) ≈ 120 instructions
- RVV: 1 vsetvl + 1 load + ~17 vector ops + 1 store ≈ 20 instructions
- **~6x instruction reduction**

## Performance Impact

`MlasComputeLogistic` is 8.29% of YOLO inference CPU time (perf analysis).
Vectorizing it also accelerates `QuickGelu` (7.97% CPU time), which calls
`MlasComputeLogistic` internally.

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

Modifies `onnxruntime/core/mlas/lib/logistic.cpp`:
- Adds `#include "rvv_compute_logistic.inl"`
- Updates `MlasComputeLogistic` to dispatch to RVV path on RISC-V targets
- Adds `MLAS_TARGET_RISCV` detection for platform dispatch

## References

- Scalar implementation: `onnxruntime/core/mlas/lib/logistic.cpp`
- QuickGelu (consumer): `onnxruntime/contrib_ops/cpu/activations.h`
- Perf analysis: `../../yolo/data/perf/yolo11n_bananapi_k1_rv64gcv_scalar_20260424_analysis.md`
