// rvv_sgemm_pack_b16.inl — B matrix 16-column packing for VL=16 SGEMM kernel
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/riscv64/SgemmPackB16.cpp  (production build)
//   - test.cpp                                             (correctness test)
//
// This packs B matrix columns into 16-element contiguous blocks for optimal
// RVV vector load (vle32.v) in the VL=16 GEMM kernel.
//
// Layout: For each 16-column block, rows are packed contiguously:
//   D[0..15]   = B_row_0[col0..col15]
//   D[16..31]  = B_row_1[col0..col15]
//   ...
//   D[CountK*16-16..CountK*16-1] = B_row_CountK-1[col0..col15]

#include <cassert>
#include <algorithm>

#if defined(MLAS_TARGET_RISCV)

inline void
MLASCALL
MlasSgemmPackB16(
    float* D,
    const float* B,
    size_t ldb,
    size_t CountX,
    size_t CountY
    )
{
    //
    // Pack 16 columns at a time (matches VL=16).
    //
    while (CountX >= 16) {

        const float* b = B;
        size_t y = CountY;

        do {
            //
            // Copy 16 contiguous floats from B row to packed buffer.
            // Layout: [col0..col15] per row, 16 floats per row.
            //
            std::copy_n(b, 16, D);

            D += 16;
            b += ldb;
            y--;

        } while (y > 0);

        B += 16;
        CountX -= 16;
    }

    //
    // Handle remaining columns (< 16) with zero-padding.
    //
    if (CountX > 0) {

        size_t y = CountY;

        do {

            std::fill_n(D, 16, 0.0f);
            std::copy_n(B, CountX, D);

            D += 16;
            B += ldb;
            y--;

        } while (y > 0);
    }
}

#endif // MLAS_TARGET_RISCV
