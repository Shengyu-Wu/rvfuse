// test.cpp — correctness test for RVV MlasLogisticKernel (sigmoid activation)
//
// Compares the RVV vectorized logistic (sigmoid) function against the scalar
// reference on deterministic pseudo-random data.
//
// Build (rv64gcv):
//   riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
//       -march=rv64gcv -mabi=lp64d \
//       -D__riscv_v_intrinsic \
//       test.cpp -o test -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Logistic constants (same as MlasLogisticConstants in MLAS)
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

// ---------------------------------------------------------------------------
// Scalar (generic) reference — matches MlasLogisticKernel in logistic.cpp
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void MlasLogisticKernel_generic(
    const float* Input,
    float* Output,
    size_t N)
{
    const LogisticConstants C;

    while (N >= 4) {
        for (int i = 0; i < 4; i++) {
            float Value = Input[i];

            // Clamp to [-18, 18] (NaN-safe)
            float v_tmp = (Value < C.LowerRange) ? C.LowerRange : Value;
            Value = (v_tmp > C.UpperRange) ? C.UpperRange : v_tmp;

            float ValueSquared = Value * Value;

            float p;
            p = ValueSquared * C.alpha_9 + C.alpha_7;
            p = p * ValueSquared + C.alpha_5;
            p = p * ValueSquared + C.alpha_3;
            p = p * ValueSquared + C.alpha_1;
            p = p * Value;

            float q;
            q = ValueSquared * C.beta_10 + C.beta_8;
            q = q * ValueSquared + C.beta_6;
            q = q * ValueSquared + C.beta_4;
            q = q * ValueSquared + C.beta_2;
            q = q * ValueSquared + C.beta_0;

            Output[i] = std::clamp((p / q) + 0.5f, 0.0f, 1.0f);
        }
        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float Value = *Input++;

        float v_tmp = (Value < C.LowerRange) ? C.LowerRange : Value;
        Value = (v_tmp > C.UpperRange) ? C.UpperRange : v_tmp;

        float ValueSquared = Value * Value;

        float p;
        p = ValueSquared * C.alpha_9 + C.alpha_7;
        p = p * ValueSquared + C.alpha_5;
        p = p * ValueSquared + C.alpha_3;
        p = p * ValueSquared + C.alpha_1;
        p = p * Value;

        float q;
        q = ValueSquared * C.beta_10 + C.beta_8;
        q = q * ValueSquared + C.beta_6;
        q = q * ValueSquared + C.beta_4;
        q = q * ValueSquared + C.beta_2;
        q = q * ValueSquared + C.beta_0;

        *Output++ = std::clamp((p / q) + 0.5f, 0.0f, 1.0f);
        N -= 1;
    }
}

// ---------------------------------------------------------------------------
// RVV vectorized implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_compute_logistic.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;
static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}
static float rng_float() {
    return (float)(rng_next()) / 32767.0f - 1.0f;
}
// Generate values in a wider range to cover the nonlinear regime
static float rng_float_wide() {
    return ((float)(rng_next()) / 32767.0f - 1.0f) * 25.0f;  // range [-25, 25)
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(size_t N, const char* label) {
    std::vector<float> input(N);
    std::vector<float> output_rvv(N);
    std::vector<float> output_scalar(N);

    for (size_t i = 0; i < N; i++) {
        input[i] = rng_float_wide();
    }

    // Run scalar reference
    MlasLogisticKernel_generic(input.data(), output_scalar.data(), N);

    // Run RVV
#if defined(__riscv_v_intrinsic)
    MlasLogisticKernel_rvv(input.data(), output_rvv.data(), N);
#else
    std::copy(output_scalar.begin(), output_scalar.end(), output_rvv.begin());
#endif

    // Compare
    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-6f;  // tight tolerance for polynomial approx

    for (size_t i = 0; i < N; i++) {
        float expected = output_scalar[i];
        float actual = output_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: input=%f expected=%.10f actual=%.10f diff=%e\n",
                       i, input[i], expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-30s N=%-6zu max_err=%.2e %s\n",
           label, N, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

int main() {
    printf("=== MlasLogisticKernel: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;
    size_t sizes[] = {1, 3, 4, 7, 8, 16, 31, 64, 100, 127, 256, 1000, 4096};

    for (size_t N : sizes) {
        char label[32];
        snprintf(label, sizeof(label), "N=%zu", N);
        failures += run_test(N, label);
    }

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
