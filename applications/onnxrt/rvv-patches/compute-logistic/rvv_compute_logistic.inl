// rvv_compute_logistic.inl — RVV implementation of MlasLogisticKernel
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/logistic.cpp  (via patch, production build)
//   - test.cpp                                (correctness test)
//
// Algorithm: Polynomial approximation of the sigmoid (logistic) function,
// same as Eigen's implementation used by MLAS.
//
//   sigmoid(x) = 1 / (1 + exp(-x))
//
// Clamp input to [-18, 18], then compute using rational polynomial:
//   p(x) = x * (alpha_1 + x^2 * (alpha_3 + x^2 * (alpha_5 + x^2 * (alpha_7 + x^2 * alpha_9))))
//   q(x) = beta_0 + x^2 * (beta_2 + x^2 * (beta_4 + x^2 * (beta_6 + x^2 * (beta_8 + x^2 * beta_10))))
//   result = clamp(p(x) / q(x) + 0.5, 0.0, 1.0)

#include <cassert>

#if defined(__riscv_v_intrinsic)
#include <riscv_vector.h>
#endif

// ---------------------------------------------------------------------------
// Logistic constants (same as MlasLogisticConstants in logistic.cpp)
// ---------------------------------------------------------------------------
struct LogisticConstants {
    float LowerRange = -18.0f;
    float UpperRange = 18.0f;
    float alpha_9  = 4.37031012579801e-11f;
    float alpha_7  = 1.15627324459942e-07f;
    float alpha_5  = 6.08574864600143e-05f;
    float alpha_3  = 8.51377133304701e-03f;
    float alpha_1  = 2.48287947061529e-01f;
    float beta_10  = 6.10247389755681e-13f;
    float beta_8   = 5.76102136993427e-09f;
    float beta_6   = 6.29106785017040e-06f;
    float beta_4   = 1.70198817374094e-03f;
    float beta_2   = 1.16817656904453e-01f;
    float beta_0   = 9.93151921023180e-01f;
    float one_half = 0.5f;
};

// =============================================================================
// RVV implementation: logistic (sigmoid) activation
// =============================================================================
// For VLEN=256 (VL=8 at SEW=32, LMUL=1), processes 8 elements per vector op.
// For VLEN=128 (VL=4), processes 4.
//
// Ported from the scalar MlasLogisticKernel in logistic.cpp.
// Vectorized using RVV float32 intrinsics with vsetvl tail handling.

#if defined(__riscv_v_intrinsic)
inline void MlasLogisticKernel_rvv(
    const float* Input,
    float* Output,
    size_t N)
{
    const LogisticConstants C;
    size_t vl;

    for (size_t i = 0; i < N; ) {
        vl = __riscv_vsetvl_e32m1(N - i);

        //
        // Load input and clamp to [-18, 18].
        //
        vfloat32m1_t Value = __riscv_vle32_v_f32m1(Input + i, vl);
        Value = __riscv_vfmax_vf_f32m1(Value, C.LowerRange, vl);
        Value = __riscv_vfmin_vf_f32m1(Value, C.UpperRange, vl);

        //
        // Compute ValueSquared = Value * Value.
        //
        vfloat32m1_t ValueSquared = __riscv_vfmul_vv_f32m1(Value, Value, vl);

        //
        // Compute p(x) polynomial (odd powers only, multiplied by x at the end).
        // p = alpha_9
        // p = p * x^2 + alpha_7
        // p = p * x^2 + alpha_5
        // p = p * x^2 + alpha_3
        // p = p * x^2 + alpha_1
        // p = p * x
        //
        vfloat32m1_t p = __riscv_vfmv_v_f_f32m1(C.alpha_9, vl);
        p = __riscv_vfmacc_vf_f32m1(p, C.alpha_7, ValueSquared, vl);
        p = __riscv_vfmacc_vf_f32m1(p, C.alpha_5, ValueSquared, vl);
        p = __riscv_vfmacc_vf_f32m1(p, C.alpha_3, ValueSquared, vl);
        p = __riscv_vfmacc_vf_f32m1(p, C.alpha_1, ValueSquared, vl);
        p = __riscv_vfmul_vv_f32m1(p, Value, vl);

        //
        // Compute q(x) polynomial (even powers).
        // q = beta_10
        // q = q * x^2 + beta_8
        // q = q * x^2 + beta_6
        // q = q * x^2 + beta_4
        // q = q * x^2 + beta_2
        // q = q * x^2 + beta_0
        //
        vfloat32m1_t q = __riscv_vfmv_v_f_f32m1(C.beta_10, vl);
        q = __riscv_vfmacc_vf_f32m1(q, C.beta_8, ValueSquared, vl);
        q = __riscv_vfmacc_vf_f32m1(q, C.beta_6, ValueSquared, vl);
        q = __riscv_vfmacc_vf_f32m1(q, C.beta_4, ValueSquared, vl);
        q = __riscv_vfmacc_vf_f32m1(q, C.beta_2, ValueSquared, vl);
        q = __riscv_vfmacc_vf_f32m1(q, C.beta_0, ValueSquared, vl);

        //
        // Compute result = clamp(p/q + 0.5, 0.0, 1.0).
        //
        vfloat32m1_t result = __riscv_vfdiv_vv_f32m1(p, q, vl);
        result = __riscv_vfadd_vf_f32m1(result, C.one_half, vl);
        result = __riscv_vfmax_vf_f32m1(result, 0.0f, vl);
        result = __riscv_vfmin_vf_f32m1(result, 1.0f, vl);

        __riscv_vse32_v_f32m1(Output + i, result, vl);
        i += vl;
    }
}
#endif // __riscv_v_intrinsic
