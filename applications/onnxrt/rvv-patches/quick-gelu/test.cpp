// test.cpp — correctness test for RVV QuickGelu alpha scaling
//
// QuickGelu(x) = x * sigmoid(alpha * x)
//
// Tests the alpha pre-scaling step (RVV vs scalar).
// The downstream sigmoid and multiply are tested separately in
// compute-logistic and eltwise-mul packages.
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
#include <vector>

// ---------------------------------------------------------------------------
// Scalar reference
// ---------------------------------------------------------------------------
#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void QuickGeluAlphaScale_generic(
    const float* input,
    float* output,
    size_t count,
    float alpha)
{
    for (size_t i = 0; i < count; i++) {
        output[i] = input[i] * alpha;
    }
}

// ---------------------------------------------------------------------------
// RVV implementation
// ---------------------------------------------------------------------------
#include "rvv_quick_gelu.inl"

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

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(size_t N, float alpha, const char* label) {
    std::vector<float> input(N);
    std::vector<float> output_rvv(N);
    std::vector<float> output_scalar(N);

    for (size_t i = 0; i < N; i++) input[i] = rng_float();

    QuickGeluAlphaScale_generic(input.data(), output_scalar.data(), N, alpha);

#if defined(__riscv_v_intrinsic)
    QuickGeluAlphaScale_rvv(input.data(), output_rvv.data(), N, alpha);
#else
    std::copy(output_scalar.begin(), output_scalar.end(), output_rvv.begin());
#endif

    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-7f;

    for (size_t i = 0; i < N; i++) {
        float expected = output_scalar[i];
        float actual = output_rvv[i];
        float err = std::fabs(expected - actual);
        if (err > max_err) max_err = err;
        if (err > tolerance) {
            if (failures < 5) {
                printf("  MISMATCH [%zu]: input=%f expected=%f actual=%f diff=%e\n",
                       i, input[i], expected, actual, err);
            }
            failures++;
        }
    }

    printf("  %-35s N=%-6zu alpha=%.3f max_err=%.2e %s\n",
           label, N, alpha, max_err, failures == 0 ? "PASS" : "FAIL");
    return failures;
}

int main() {
    printf("=== QuickGelu Alpha Scale: RVV vs Scalar correctness test ===\n\n");

    int failures = 0;
    size_t sizes[] = {1, 3, 4, 7, 8, 16, 31, 64, 100, 127, 256, 1000, 4096};
    float alphas[] = {1.702f, 1.0f, 2.0f, 0.5f};

    for (float alpha : alphas) {
        for (size_t N : sizes) {
            char label[48];
            snprintf(label, sizeof(label), "alpha=%.3f N=%zu", alpha, N);
            failures += run_test(N, alpha, label);
        }
    }

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
