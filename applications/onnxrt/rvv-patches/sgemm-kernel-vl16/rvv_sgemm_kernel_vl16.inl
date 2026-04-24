// rvv_sgemm_kernel_vl16.inl — RVV implementation of MlasSgemmKernel (VL=16, 512-bit)
//
// Single source of truth. Included by:
//   - onnxruntime/core/mlas/lib/riscv64/SgemmKernelRvv512.cpp  (production build)
//   - test.cpp                                                 (correctness test)
//
// This kernel targets 512-bit vector registers (VLEN=512, SEW=32, LMUL=1, VL=16).
// It uses 16-column B packing for optimal memory access, processing 16 output
// elements per vector instruction (4x the VL=4 kernel).
//
// Prerequisites:
//   - VLEN_512 must be defined (for MLAS_RVV_VLEN_512)
//   - mlasi.h must be included first
//   - rvv_sgemm_pack_b16.inl must be included for MlasSgemmPackB16

#include <cassert>

#if defined(MLAS_RVV_VLEN_512) && defined(__riscv_v)
#include <riscv_vector.h>

//
// Inner kernel: computes a 16x1 or 16x2 output block using RVV intrinsics.
// VL=16 processes 16 packed B columns per vector instruction, providing
// 4x higher compute density than the VL=4 kernel.
//
// Per K iteration (unrolled by 2):
//   VL=4 kernel:  4 flw + 2 vle32.v (4 elem) + 4 vfmacc.vf = 10 ops, 4 output cols
//   VL=16 kernel: 4 flw + 2 vle32.v (16 elem) + 2 vfmacc.vf = 8 ops, 16 output cols
//   Ratio: 16/4 = 4x more output per iteration with fewer ops
//

template<bool ZeroMode, bool ProcessTwoRows>
inline size_t
MlasSgemmKernelRvv512Impl(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha
    )
{
    constexpr size_t kVlen = 16;  // 512-bit VLEN, SEW=32, LMUL=1
    const size_t vl = __riscv_vsetvl_e32m1(kVlen);

    //
    // Process 16-column output blocks.
    //
    do {
        //
        // Clear the vector accumulators (16-element vectors).
        //
        vfloat32m1_t v_acc_r0 = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t v_acc_r1 = __riscv_vfmv_v_f_f32m1(0.0f, vl);

        const float* a = A;
        const float* b = B;
        size_t k = CountK;

        //
        // Vectorized K loop, unrolled by 2.
        //
        while (k >= 2) {

            //
            // Load A elements (scalar — broadcast into vector FMA).
            //
            float a0_r0 = a[0];
            float a1_r0 = a[1];

            float a0_r1 = 0.0f;
            float a1_r1 = 0.0f;
            if (ProcessTwoRows) {
                a0_r1 = a[lda];
                a1_r1 = a[lda + 1];
            }

            //
            // Load B[0..15] (K element 0) and FMA.
            // B is packed as 16 contiguous floats per K row.
            //
            vfloat32m1_t v_b0 = __riscv_vle32_v_f32m1(b, vl);
            v_acc_r0 = __riscv_vfmacc_vf_f32m1(v_acc_r0, a0_r0, v_b0, vl);
            if (ProcessTwoRows) {
                v_acc_r1 = __riscv_vfmacc_vf_f32m1(v_acc_r1, a0_r1, v_b0, vl);
            }

            //
            // Load B[16..31] (K element 1) and FMA.
            // Offset 16 floats = next K row in packed buffer.
            //
            vfloat32m1_t v_b1 = __riscv_vle32_v_f32m1(b + 16, vl);
            v_acc_r0 = __riscv_vfmacc_vf_f32m1(v_acc_r0, a1_r0, v_b1, vl);
            if (ProcessTwoRows) {
                v_acc_r1 = __riscv_vfmacc_vf_f32m1(v_acc_r1, a1_r1, v_b1, vl);
            }

            a += 2;
            b += 32;    // B packing stride: 2 × 16 floats per K-pair
            k -= 2;
        }

        //
        // Handle remaining K element (odd CountK).
        //
        if (k > 0) {
            float a0_r0 = a[0];
            float a0_r1 = 0.0f;
            if (ProcessTwoRows) {
                a0_r1 = a[lda];
            }

            vfloat32m1_t v_b0 = __riscv_vle32_v_f32m1(b, vl);
            v_acc_r0 = __riscv_vfmacc_vf_f32m1(v_acc_r0, a0_r0, v_b0, vl);
            if (ProcessTwoRows) {
                v_acc_r1 = __riscv_vfmacc_vf_f32m1(v_acc_r1, a0_r1, v_b0, vl);
            }
        }

        //
        // Multiply accumulators by alpha.
        //
        v_acc_r0 = __riscv_vfmul_vf_f32m1(v_acc_r0, alpha, vl);
        if (ProcessTwoRows) {
            v_acc_r1 = __riscv_vfmul_vf_f32m1(v_acc_r1, alpha, vl);
        }

        //
        // Store output block.
        //
        if (CountN >= 16) {

            //
            // Store the full 16-element output block.
            //
            if (!ZeroMode) {
                vfloat32m1_t v_c = __riscv_vle32_v_f32m1(C, vl);
                v_acc_r0 = __riscv_vfadd_vv_f32m1(v_acc_r0, v_c, vl);
            }
            __riscv_vse32_v_f32m1(C, v_acc_r0, vl);

            if (ProcessTwoRows) {
                if (!ZeroMode) {
                    vfloat32m1_t v_c = __riscv_vle32_v_f32m1(C + ldc, vl);
                    v_acc_r1 = __riscv_vfadd_vv_f32m1(v_acc_r1, v_c, vl);
                }
                __riscv_vse32_v_f32m1(C + ldc, v_acc_r1, vl);
            }

            //
            // Advance to next 16-column block.
            // B stride: CountK × 16 floats (packed layout).
            //
            B += CountK * 16;
            C += 16;
            CountN -= 16;

        } else {

            //
            // Store the partial output block (CountN < 16).
            // Extract scalars from vector for partial store.
            //
            float row0[16], row1[16];
            __riscv_vse32_v_f32m1(row0, v_acc_r0, vl);
            if (ProcessTwoRows) {
                __riscv_vse32_v_f32m1(row1, v_acc_r1, vl);
            }

            for (size_t i = 0; i < CountN; i++) {
                if (!ZeroMode) {
                    row0[i] = row0[i] + C[i];
                }
                C[i] = row0[i];

                if (ProcessTwoRows) {
                    if (!ZeroMode) {
                        row1[i] = row1[i] + C[ldc + i];
                    }
                    C[ldc + i] = row1[i];
                }
            }

            break;
        }

    } while (CountN > 0);

    return ProcessTwoRows ? 2 : 1;
}

//
// Dispatch entry point matching MLAS_GEMM_FLOAT_KERNEL signature.
//
inline size_t
MLASCALL
MlasSgemmKernelRvv512(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountM,
    size_t CountN,
    size_t lda,
    size_t ldc,
    float alpha,
    bool ZeroMode
    )
{
    if (ZeroMode) {
        if (CountM >= 2) {
            return MlasSgemmKernelRvv512Impl<true, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
        } else {
            return MlasSgemmKernelRvv512Impl<true, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
        }
    } else {
        if (CountM >= 2) {
            return MlasSgemmKernelRvv512Impl<false, true>(A, B, C, CountK, CountN, lda, ldc, alpha);
        } else {
            return MlasSgemmKernelRvv512Impl<false, false>(A, B, C, CountK, CountN, lda, ldc, alpha);
        }
    }
}

#endif // MLAS_RVV_VLEN_512 && __riscv_v
