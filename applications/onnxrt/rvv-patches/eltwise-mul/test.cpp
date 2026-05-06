// test.cpp — correctness test for RVV MlasEltwiseMul<float>
//
// Compares the RVV vectorized element-wise multiply against scalar reference.
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

// ---------------------------------------------------------------------------
// Scalar reference — matches MlasEltwiseMul<float> in eltwise.cpp
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void MlasEltwiseMulF32_generic(
    const float* left,
    const float* right,
    float* output,
    size_t N)
{
    for (size_t i = 0; i < N; i++) {
        output[i] = left[i] * right[i];
    }
}

// ---------------------------------------------------------------------------
// RVV implementation — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_eltwise_mul.inl"

// ---------------------------------------------------------------------------
// Pseudo-random data generator
// ---------------------------------------------------------------------------
static uint32_t rng_state = 42;
static uint32_t rng_next() {
    rng_state = rng_state * 1103515245 + 12345;
    return (rng_state >> 16) & 0x7FFF;
}
static float rng_float() {
    return (float)(rng_next()) / 32767.0f - 1.0f;  // range [-1, 1)
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(size_t N, const char* label) {
    std::vector<float> left(N), right(N);
    std::vector<float> output_rvv(N);
    std::vector<float> output_scalar(N);

    for (size_t i = 0; i < N; i++) {
        left[i] = rng_float();
        right[i] = rng_float();
    }

    // Run scalar reference
    MlasEltwiseMulF32_generic(left.data(), right.data(), output_scalar.data(), N);

    // Run RVV
#if defined(__riscv_v_intrinsic)
    MlasEltwiseMulF32_rvv(left.data(), right.data(), output_rvv.data(), N);
#else
    std::copy(output_scalar.begin(), output_scalar.end(), output_rvv.begin());
#endif

    // Compare (element-wise multiply is exact for FP32, so use tight tolerance)
    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-7f;  // FP32 multiply is exact

    for (size_t i = 0; i < N; i++) {
        float expected = output_scalar[i];
        float actual = output_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: left=%f right=%f expected=%f actual=%f diff=%e\n",
                       i, left[i], right[i], expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-30s N=%-6zu max_err=%.2e %s\n",
           label, N, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

int main() {
    printf("=== MlasEltwiseMul: RVV vs Scalar correctness test ===\n\n");

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
