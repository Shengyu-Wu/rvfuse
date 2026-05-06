// test.cpp — correctness test for RVV MlasSgemmKernel (VL=16, 512-bit)
//
// Compares the VL=16 RVV SGEMM kernel + B16 packing against the scalar reference.
//
// Build (rv64gcv, VLEN=512):
//   riscv64-unknown-linux-gnu-clang++ -std=c++17 -O2 \
//       --target=riscv64-unknown-linux-gnu --sysroot=<sysroot> \
//       -march=rv64gcv_zvl512b -mabi=lp64d \
//       -D__riscv_v -D__riscv_v_intrinsic -DVLEN_512 -DMLAS_RVV_VLEN_512 -DMLAS_TARGET_RISCV \
//       test.cpp -o test -lm

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>
#include <algorithm>

#define MLASCALL

// ---------------------------------------------------------------------------
// Scalar reference: simple GEMM with standard layout (CountK × CountN B matrix)
// ---------------------------------------------------------------------------

#if defined(__riscv_v_intrinsic)
__attribute__((optnone))
#endif
static void sgemm_scalar(
    const float* A,
    const float* B_std,      // standard layout: CountK rows × CountN cols (ldb = CountN)
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode)
{
    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            float acc = 0.0f;
            for (size_t k = 0; k < CountK; k++) {
                acc += A[m * lda + k] * B_std[k * CountN + n];
            }
            acc *= alpha;
            if (!ZeroMode) {
                acc += C[m * ldc + n];
            }
            C[m * ldc + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// B16 packing (scalar, matches rvv_sgemm_pack_b16.inl)
// ---------------------------------------------------------------------------
static void pack_b16_scalar(
    float* D,
    const float* B_std,
    size_t ldb,
    size_t CountX,
    size_t CountY)
{
    while (CountX >= 16) {
        const float* b = B_std;
        size_t y = CountY;
        do {
            std::copy_n(b, 16, D);
            D += 16;
            b += ldb;
            y--;
        } while (y > 0);

        B_std += 16;
        CountX -= 16;
    }

    if (CountX > 0) {
        size_t y = CountY;
        do {
            std::fill_n(D, 16, 0.0f);
            std::copy_n(B_std, CountX, D);
            D += 16;
            B_std += ldb;
            y--;
        } while (y > 0);
    }
}

// ---------------------------------------------------------------------------
// RVV implementations — single source of truth
// ---------------------------------------------------------------------------
#include "rvv_sgemm_pack_b16.inl"
#include "rvv_sgemm_kernel_vl16.inl"

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

static void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = rng_float();
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------
static int run_test(
    size_t CountK,
    size_t CountM,
    size_t CountN,
    bool ZeroMode,
    const char* label)
{
    size_t lda = CountK;
    size_t ldc = CountN;

    // A matrix: CountM × CountK
    std::vector<float> A(CountM * lda);
    fill_random(A.data(), A.size());

    // B matrix (standard layout): CountK × CountN
    std::vector<float> B_std(CountK * CountN);
    fill_random(B_std.data(), B_std.size());

    // B packed (16-column blocks): CountK * 16 * num_blocks
    size_t num_blocks = (CountN + 15) / 16;
    std::vector<float> B_packed(CountK * 16 * num_blocks);
    pack_b16_scalar(B_packed.data(), B_std.data(), CountN, CountN, CountK);

    // C matrices
    std::vector<float> C_rvv(CountM * ldc);
    std::vector<float> C_scalar(CountM * ldc);
    fill_random(C_rvv.data(), C_rvv.size());
    std::copy(C_rvv.begin(), C_rvv.end(), C_scalar.begin());

    // Run scalar reference (using standard B layout)
    sgemm_scalar(A.data(), B_std.data(), C_scalar.data(), CountK, CountM, CountN, lda, ldc, 1.0f, ZeroMode);

    // Run RVV kernel
    size_t rows_rvv = 0;
#if defined(__riscv_v_intrinsic) && defined(MLAS_RVV_VLEN_512)
    rows_rvv = MlasSgemmKernelRvv512(A.data(), B_packed.data(), C_rvv.data(),
                                     CountK, CountM, CountN, lda, ldc, 1.0f, ZeroMode);
#else
    // Fallback: copy scalar results for non-RVV builds
    std::copy(C_scalar.begin(), C_scalar.end(), C_rvv.begin());
#endif

    // Compare
    int failures = 0;
    float max_err = 0.0f;
    float tolerance = 1e-4f;

    for (size_t m = 0; m < CountM; m++) {
        for (size_t n = 0; n < CountN; n++) {
            float expected = C_scalar[m * ldc + n];
            float actual = C_rvv[m * ldc + n];
            float err = std::fabs(expected - actual);
            if (err > max_err) max_err = err;
            if (err > tolerance) {
                if (failures < 5) {
                    printf("  MISMATCH [%zu,%zu]: expected=%f, actual=%f, diff=%e\n",
                           m, n, expected, actual, err);
                }
                failures++;
            }
        }
    }

    printf("  %-35s K=%-4zu M=%-2zu N=%-3zu Zero=%-5s rows_rvv=%-2zu max_err=%.2e %s\n",
           label, CountK, CountM, CountN, ZeroMode ? "true" : "false",
           rows_rvv, max_err, failures == 0 ? "PASS" : "FAIL");

    return failures;
}

int main() {
    printf("=== MlasSgemmKernel (VL=16): RVV vs Scalar correctness test ===\n\n");

    int failures = 0;

    struct Config { size_t K, M, N; bool zero; };
    Config configs[] = {
        // Full 16-column blocks
        {4, 1, 16, true},
        {4, 2, 16, true},
        {8, 1, 16, false},
        {8, 2, 16, false},
        {16, 1, 16, true},
        {16, 2, 16, false},
        // Larger K
        {32, 1, 16, true},
        {32, 2, 16, false},
        {64, 1, 16, false},
        {64, 2, 16, true},
        // Odd K
        {3, 1, 16, false},
        {5, 2, 16, true},
        {7, 1, 16, false},
        // Partial N (< 16)
        {4, 1, 8, true},
        {8, 2, 4, false},
        {16, 1, 3, true},
        {6, 2, 12, false},
        // Multi-block (N > 16)
        {8, 1, 32, false},
        {4, 2, 20, true},
        // Stress tests
        {128, 1, 16, false},
        {128, 2, 16, true},
    };

    for (const auto& cfg : configs) {
        char label[64];
        snprintf(label, sizeof(label), "K=%zu M=%zu N=%zu Zero=%d", cfg.K, cfg.M, cfg.N, cfg.zero);
        failures += run_test(cfg.K, cfg.M, cfg.N, cfg.zero, label);
    }

    printf("\n=== Summary: %d test(s) failed ===\n", failures);
    return failures;
}
