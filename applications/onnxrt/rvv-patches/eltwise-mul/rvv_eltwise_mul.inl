// rvv_eltwise_mul.inl — RVV implementation of MlasEltwiseMul<float>
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/eltwise.cpp  (via patch, production build)
//   - test.cpp                               (correctness test)
//
// Element-wise multiply: output[i] = left[i] * right[i]
//
// RVV vectorization: uses vle32.v for vector loads and vfmul.vv for
// vector multiply. Tail elements handled via vsetvl.

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// =============================================================================
// RVV implementation: element-wise float multiply
// =============================================================================

#if defined(__riscv_v_intrinsic)
inline void MlasEltwiseMulF32_rvv(
    const float* left,
    const float* right,
    float* output,
    size_t N)
{
    size_t vl;

    for (size_t i = 0; i < N; ) {
        vl = __riscv_vsetvl_e32m1(N - i);

        vfloat32m1_t v_left  = __riscv_vle32_v_f32m1(left + i, vl);
        vfloat32m1_t v_right = __riscv_vle32_v_f32m1(right + i, vl);
        vfloat32m1_t v_result = __riscv_vfmul_vv_f32m1(v_left, v_right, vl);

        __riscv_vse32_v_f32m1(output + i, v_result, vl);
        i += vl;
    }
}
#endif // __riscv_v_intrinsic
