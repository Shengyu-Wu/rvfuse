// rvv_quick_gelu.inl — RVV helper for QuickGelu alpha scaling
//
// Single source of truth. Included by:
//   - onnxruntime/contrib_ops/cpu/activations.h  (via patch, production build)
//   - test.cpp                                    (correctness test)
//
// QuickGelu(x) = x * sigmoid(alpha * x)
//
// The sigmoid and element-wise multiply are already RVV-vectorized in
// MlasComputeLogistic and MlasEltwiseMul. This file provides the alpha
// pre-scaling step: output[i] = input[i] * alpha.
//
// When alpha == 1.0, this step is skipped entirely.

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

#if defined(__riscv_v_intrinsic)
//
// Vectorized alpha scaling: output[i] = input[i] * alpha
//
inline void QuickGeluAlphaScale_rvv(
    const float* input,
    float* output,
    size_t count,
    float alpha)
{
    size_t vl;
    for (size_t i = 0; i < count; ) {
        vl = __riscv_vsetvl_e32m1(count - i);
        vfloat32m1_t v = __riscv_vle32_v_f32m1(input + i, vl);
        v = __riscv_vfmul_vf_f32m1(v, alpha, vl);
        __riscv_vse32_v_f32m1(output + i, v, vl);
        i += vl;
    }
}
#endif // __riscv_v_intrinsic
