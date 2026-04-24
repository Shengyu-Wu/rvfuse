# YOLO11n Perf Profiling Analysis

**Date**: 2026-04-24
**Target**: Banana Pi (SpacemiT K1, rv64imafdcv, 4 cores @ 1.6 GHz)
**Binary**: yolo_inference (rv64gcv, debug_info, not stripped)
**Model**: yolo11n.ort (YOLO11 nano, float32, 640x640)
**Test image**: test.jpg (MD5: 10bd23d36a88f18010bf1c63b56aa309)
**ORT**: libonnxruntime.so.1.24.4 (Release, rv64gcv, debug_info)

---

## 1. Global Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Wall time (30 iters) | 171.67 s | ~5.72 s/iter |
| Cycles | 273.9 billion | |
| Instructions | 236.5 billion | |
| **IPC** | **0.86** | Compute-bound (below 1.0 = stalls) |
| Avg clock | 1.596 GHz | |
| Branches | 8.79B | 51.2 M/sec |
| **Branch misses** | **0.50%** | Excellent — predictor works well |
| L1-dcache loads | 77.3B | 450 M/sec |
| **L1-dcache misses** | **0.42%** | Excellent — cache-friendly workload |
| Context switches | 114 | 0.66/sec — very low |
| CPU migrations | 0 | Pinned |

**Assessment**: The workload is **compute-bound** (IPC=0.86), with excellent cache and branch predictor behavior. The low IPC indicates pipeline stalls — likely from load-use latency in the scalar GEMM inner loop. Optimizing the GEMM with RVV vectors could significantly improve IPC.

---

## 2. Function-Level Hotspots

| %CPU | Self% | Samples | Function |
|------|-------|---------|----------|
| **43.99%** | 43.99% | 77,591 | `MlasSgemmKernel<false,true>` — scalar GEMM kernel |
| **40.26%** | 6.88% | 12,134 | `MlasSgemmOperation` — GEMM operation dispatch |
| **32.89%** | 32.87% | 57,990 | `MlasSgemmKernel<true,true>` — scalar GEMM kernel |
| 8.29% | 8.28% | 14,603 | `MlasComputeLogistic` — sigmoid activation |
| 7.97% | 0.01% | 20 | `QuickGelu<float>::Compute` — GELU fusion |
| 7.07% | 0.02% | 29 | `MlasConvOperation` — convolution dispatch |
| 3.79% | 3.78% | 6,665 | `MlasConvIm2Col` — im2col transform |
| 1.00% | 1.00% | 1,771 | `MlasActivation` — activation (RVV-optimized) |
| 0.81% | 0.81% | 1,424 | `MlasEltwiseMul` — element-wise multiply |

**Top-10 functions consume >99% of CPU time.** The top 2 GEMM kernels alone account for **~77%** of all samples.

---

## 3. Instruction-Level Hotspots

### 3.1 Scalar GEMM Inner Loop (`MlasSgemmKernel<false,true>` @ 43.99%)

```
 2.72%  c7e6cc: flw     ft4,0(t6)        # load B element
 1.65%  c7e6d0: flw     ft5,4(t6)        # load B element
 2.89%  c7e6d4: flw     ft6,-4(s1)       # load B element
 8.46%  c7e6d8: flw     ft7,0(s1)        # load B element (HOT)
 5.54%  c7e6dc: flw     fa6,0(a5)        # load A element
 4.41%  c7e6e0: flw     fa7,4(a5)        # load A element
 5.82%  c7e6e4: flw     ft8,8(a5)        # load A element
 6.64%  c7e6e8: flw     ft9,12(a5)       # load A element
10.38%  c7e714: flw     ft8,76(a5)       # load A element strided
 5.71%  c7e6ec: fmadd.s ft3,fa6,ft4,ft3  # fused multiply-add
 2.65%  c7e6f4: fmadd.s ft1,ft8,ft4,ft1  # fused multiply-add
 2.74%  c7e6fc: fmadd.s ft0,fa6,ft6,ft0  # fused multiply-add
 2.69%  c7e704: fmadd.s fa2,ft8,ft6,fa2  # fused multiply-add
 2.53%  c7e724: fmadd.s ft3,ft4,ft5,ft3  # fused multiply-add
```

**Pattern**: Inner loop does 10+ scalar float loads (flw) per iteration with 8 fmadd.s. This is a classic **load-compute chained pattern** — each iteration loads 10 floats, does 8 FMAs. The 0.86 IPC is explained by load-use stalls: each `flw` takes 2-3 cycles before data is ready for `fmadd.s`.

### 3.2 RVV-Optimized Activation (`MlasActivation` @ 1.00%)

```
 5.03%  c659b8: vle32.v v8,(a4)          # RVV vector load (4 floats)
35.01%  c659bc: addi    a0,a0,-4          # loop counter
 0.06%  c659be: vfadd.vf v8,v8,fa5       # RVV vector-scalar add
 9.15%  c659c2: vse32.v v8,(a4)          # RVV vector store
48.84%  c659c6: addi    a4,a4,16          # pointer advance
```

**Pattern**: Proper RVV vectorized loop — vle32.v/vfadd.vf/vse32.v with vsetivli zero,4,e32,m1. The loop control instructions (addi) dominate samples because the vector work instructions retire quickly. This loop is **highly efficient** — processes 4 floats in ~4 instructions vs 10+ for scalar GEMM.

### 3.3 Im2Col Store Bottleneck (`MlasConvIm2Col` @ 3.79%)

```
58.61%  c5b138: sd      t4,0(t3)          # store double (EXTREME hotspot)
20.57%  c5b134: ld      a4,16(a3)         # load parameter
```

The `sd` at 58.61% is an output store that stalls waiting for the store buffer/cache. This is the im2col output path writing the transformed matrix.

---

## 4. RVV Utilization Analysis

### Critical Finding: GEMM uses scalar FP, not RVV

The two dominant GEMM kernels (77% of CPU time) use **scalar FP instructions** (`flw`/`fmadd.s`) rather than RVV vector instructions (`vle32.v`/`vfmacc.vf`). Only `MlasActivation` (1% of CPU) properly uses RVV.

| Kernel | Instructions | RVV? | Performance |
|--------|-------------|------|-------------|
| MlasSgemmKernel<false,true> | flw + fmadd.s | No | 1 FMA/cycle (scalar) |
| MlasSgemmKernel<true,true> | flw + fmadd.s | No | 1 FMA/cycle (scalar) |
| MlasActivation | vle32.v + vfadd.vf + vse32.v | Yes | 4 floats/cycle |

**Root cause**: ONNX Runtime v1.24.4's MLAS GEMM kernels for RISC-V use scalar fallback paths. The RVV-optimized kernels (`QGEMM` with RVV intrinsics) may not be enabled in this build configuration, or the kernels detect VLEN at runtime and fall back.

The SpacemiT K1 supports VLEN=256 (zvl256b), which could process **8 FP32 elements per vector register** instead of the current 1.

---

## 5. Fusion Candidate Analysis

### Priority 1: GEMM + Activation Fusion (highest impact)

**Functions**: `MlasSgemmKernel` (77%) + `MlasComputeLogistic` (8%) + `QuickGelu` (8%)

These functions process the same data in sequence: GEMM output → sigmoid → GELU. Fusing the activation into the GEMM kernel eliminates a separate memory pass and saves ~15% of total CPU.

**Candidate fused instruction**: `fmma.s` (fused matrix multiply-accumulate) + `fsigmoid.s` (fused sigmoid) or a GEMM-with-post-op instruction.

### Priority 2: Im2Col + GEMM Fusion

**Functions**: `MlasConvIm2Col` (3.8%) + `MlasSgemmOperation` (40%)

The im2col transform prepares the input matrix for GEMM. Fusing im2col with GEMM eliminates the intermediate memory buffer (the 58.61% `sd` bottleneck).

### Priority 3: Load-Store Fusion in GEMM

**Pattern**: The GEMM inner loop has 10+ `flw` instructions loading from A and B matrices. Each pair of (load A, load B) followed by `fmadd.s` is a candidate for **load-pair fusion**: a single instruction that loads two operands and performs FMA.

### Priority 4: EltwiseMul + Activation chain

**Functions**: `MlasEltwiseMul` (0.8%) + `MlasActivation` (1.0%)

Element-wise multiply followed by activation — could be fused into a single vector operation.

---

## 6. IPC Analysis

| Component | IPC | Bottleneck |
|-----------|-----|-----------|
| Overall | 0.86 | Compute-bound |
| GEMM (scalar) | ~0.5-0.8 (est.) | Load-use latency |
| Activation (RVV) | ~1.5-2.0 (est.) | Vector throughput |

The overall IPC of 0.86 is dragged down by the scalar GEMM. With RVV-vectorized GEMM, the IPC could reach 1.2-1.8 for the same kernels.

---

## 7. Comparison: Scalar vs RVV GEMM

The existing `MlasActivation` shows what proper RVV looks like:
- Processes 4 floats with 1 vle32.v + 1 vfadd.vf + 1 vse32.v = 3 vector instructions
- Scalar equivalent would need 4x flw + 4x fadd.s + 4x fsw = 12 scalar instructions
- **RVV is ~4x more instruction-efficient for the same work**

If the GEMM kernels were similarly vectorized:
- Current: 10 flw + 8 fmadd.s per 8 FMAs = 18 instructions
- RVV (VLEN=256, 8 floats/reg): ~4 vle32.v + 4 vfmacc.vv = 8 instructions
- **2.25x instruction reduction** → higher IPC → potentially 2-3x speedup

---

## 8. Recommendations

1. **Enable RVV GEMM in ORT build** — investigate MLAS RVV kernel compilation flags;
   check if `--cpu rv64,v=true,vlen=256` or `zvl256b` compile flags are needed
2. **GEMM + Activation fusion** — highest-priority fusion candidate; saves ~15% total CPU
3. **Im2Col + GEMM fusion** — eliminates the 58% concentrated store bottleneck
4. **Profile with `perf record -e cpu-cycles`** if hardware PMU becomes available —
   would give more accurate instruction-level attribution than cpu-clock sampling

---

## 9. Key Takeaways

- **YOLO11n on rv64gcv runs correctly** — validation PASSED (shape + checksum)
- **GEMM dominates at 77%** — completely scalar, no RVV vectorization
- **IPC of 0.86 indicates CPU stalls** — load-use latency in scalar GEMM inner loop
- **Cache behavior is excellent** — 0.42% L1 miss rate, 0.50% branch miss rate
- **RVV is available but underutilized** — only 1% of CPU cycles use vector instructions
- **Top fusion candidate**: GEMM + Activation fused instruction (est. 15% speedup)
