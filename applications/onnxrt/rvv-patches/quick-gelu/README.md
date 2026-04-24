# quick-gelu

RVV implementation of `QuickGelu<float>::Compute` alpha pre-scaling step — RISC-V Vector Extension.

## Status

✅ Implementation complete — acceleration is achieved through three vectorized components.

## Architecture

`QuickGelu(x) = x * sigmoid(alpha * x)` is computed in three stages:

| Stage | Function | Status |
|-------|----------|--------|
| 1. Alpha scaling | `QuickGeluAlphaScale_rvv` | This package |
| 2. Sigmoid | `MlasComputeLogistic` (RVV) | `../compute-logistic/` |
| 3. Element-wise multiply | `MlasEltwiseMul` (RVV) | `../eltwise-mul/` |

When `alpha == 1.0`, stage 1 is skipped entirely (SiLU activation).

## Files

| File | Purpose |
|------|---------|
| `rvv_quick_gelu.inl` | RVV alpha scaling helper (single source of truth) |
| `patch.diff` | Patch to integrate into ONNX Runtime contrib_ops |
| `test.cpp` | Correctness test (RVV vs scalar reference) |

## Function Signature

```cpp
template <typename T>
class QuickGelu : public OpKernel {
  float alpha_;
  Status Compute(OpKernelContext* context) const override;
};
```

## Algorithm

```
for each element x:
  if alpha != 1.0:
    x = x * alpha          // Stage 1: this package (RVV vfmul.vf)
  y = sigmoid(x)           // Stage 2: MlasComputeLogistic (RVV polynomial)
  output = x * y           // Stage 3: MlasEltwiseMul (RVV vfmul.vv)
```

## RVV Vectorization

The alpha scaling is vectorized with `vfmul.vf` (vector-scalar multiply):
- Load input via `vle32.v`
- Multiply by alpha: `vfmul.vf`
- Store via `vse32.v`
- Dynamic VL via `vsetvl` for tail handling

## Performance Impact

`QuickGelu` is 7.97% of YOLO inference CPU time. With all three stages vectorized:
- Stage 1: ~6x instruction reduction for alpha scaling
- Stage 2: ~6x instruction reduction for sigmoid (see compute-logistic)
- Stage 3: ~6x instruction reduction for multiply (see eltwise-mul)

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

## Dependencies

This operator depends on:
- `compute-logistic` — RVV logistic/sigmoid kernel
- `eltwise-mul` — RVV element-wise multiply kernel

All three patches must be applied together for full QuickGelu acceleration.

## References

- ONNX Runtime implementation: `onnxruntime/contrib_ops/cpu/activations.h`
- Perf analysis: `../../yolo/data/perf/yolo11n_bananapi_k1_rv64gcv_scalar_20260424_analysis.md`
