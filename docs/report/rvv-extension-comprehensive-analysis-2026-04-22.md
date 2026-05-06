# RVV扩展指令综合分析报告（llama.cpp + YOLO）

> **数据来源**：
> - llama.cpp: `docs/report/llama_cpp_perf_q4_v_analysis.md`, `docs/report/llama.cpp/rvv-gap-analysis-gemv_q8_0_16x1_q8_0-2026-04-17.md`
> - YOLO: `docs/report/yolo/rvv-extension-analysis-report.md`, `docs/report/yolo/multi-platform-vector-comparison.md`
> - 指令延迟: `docs/reference/cx/instruction-constraints-and-latency.md`

---

## 一、热点分布对比

### 1.1 llama.cpp 推理热点

| 函数 | 占比 | 计算类型 |
|------|------|---------|
| `ggml_gemv_q4_0_16x1_q8_0` | **40.68%** | INT8量化 GEMV（decode阶段） |
| `ggml_gemv_q8_0_16x1_q8_0` | **24.06%** | INT8量化 GEMV（K-quant中间层） |
| `ggml_gemm_q4_0_16x1_q8_0` | 8.05% | INT8量化 GEMM（prefill阶段） |
| `repack_q4_0_to_q4_0_16_bl` | 11.73% | 数据重打包 |
| 其他算子 | ~15% | Norm/RoPE/Attention等 |

**GEMV/GEMM 总计**: **72.79%**（核心计算）

### 1.2 YOLO 推理热点

| 算子 | 占比 | 计算类型 |
|------|------|---------|
| Conv/GEMM | **35%** | INT8 MAC密集（量化推理） |
| Conv/GEMM | 35% | FP32 MAC密集（部分层） |
| BatchNorm | 15% | 元素级运算 |
| Activation | 10% | SiLU/ReLU等 |
| 其他 | 15% | 数据搬运/Concat |

**Conv/GEMM 总计**: **70%**（核心计算）

---

## 二、扩展指令方案汇总（区分来源应用）

### 2.1 方案总表

| 指令名称 | 来源应用 | 计算类型 | 解决的问题 | 整体收益 |
|----------|----------|----------|-----------|----------|
| **vsegdot.vv** | YOLO + llama.cpp | INT8 GEMM/GEMV | 分段规约缺失（4×i8→i32） | **YOLO: 20-26%（整体），llama.cpp: 15-25%（整体）** |
| **vdot_lane.vx** | llama.cpp | INT8 GEMV | Lane-indexed dot消除标量广播 | **llama.cpp: 15-25%（整体）** |
| **vmulacc.vv** | YOLO | FP32 GEMM | 矩阵外积（4×4 MAC） | **YOLO: 10-13%（整体）** |
| **vfmacc.vv_lane** | 通用 | FP32 MAC | Lane-indexed FMA | **YOLO: 6-16%（整体），llama.cpp: 5-8%（整体）** |
| **vunzip/vzip.vv** | YOLO | 数据重排 | 奇偶分离/合并（转置） | **YOLO: 边际收益（非热点）** |
| **vwmaccwev/wod.vv** | llama.cpp | Widening MAC | Even/Odd分离减少依赖链 | **llama.cpp: 3-5%（整体）** |

### 2.2 跨应用通用方案

以下指令对两个应用均有收益：

| 指令 | llama.cpp收益（整体） | YOLO收益（整体） | 说明 |
|------|---------------------|----------------|------|
| `vsegdot.vv` | **15-25%** | **20-26%** | 分段点积，两应用通用 |
| `vfmacc.vv_lane` / `vdot_lane.vx` | **5-8%** | **6-16%** | Lane-indexed思想通用 |

---

## 三、INT8 路径方案详解

### 3.1 核心瓶颈：分段规约缺失

**问题本质**（来自YOLO分析）：

```
x86 VNNI vpdpbusd（512-bit）：
  输入：64个int8 → 分成16组，每组4个
  输出：16个独立int32结果（并行产出）

ARM SDOT（128-bit）：
  输入：16个int8 → 分成4组，每组4个
  输出：4个独立int32结果

RVV vredsum：
  输入：VL个int8
  输出：仅1个int32标量（串行瓶颈）
```

**对GEMM数据流的影响**：

| 架构 | 每次加载产出的C元素数 | Store效率 |
|------|---------------------|----------|
| x86 VNNI | 8个int32/指令 | SIMD批量写回 |
| ARM SDOT | 4个int32/指令 | SIMD批量写回 |
| RVV（无分段规约） | 1个int32 | 串行scalar写回 |

### 3.2 方案A：vsegdot.vv（YOLO来源）

**指令定义**：

```
vsegdot.vv vd, vs1, vs2, seg=4
  输入：
    vs1 = vint8m1_t [a0 a1 a2 a3 | a4 a5 a6 a7 | ...]（直接int8向量）
    vs2 = vint8m1_t [b0 b1 b2 b3 | b4 b5 b6 b7 | ...]
  输出：
    vd = vint32mf2_t（元素数为VL/4）
    vd[0] += a0*b0 + a1*b1 + a2*b2 + a3*b3
    vd[1] += a4*b4 + a5*b5 + a6*b6 + b7*b7
```

**计算场景**：16 个独立 int32 点积，K=32，VLEN=512。

| 架构 | 单指令int32产出 | K=32迭代次数 | K维并行 |
|------|----------------|-------------|--------|
| RVV外积法 | 0（int16需拓宽） | **32次** | 1 |
| vsegdot | **16个** | **8次** | 4 |
| x86 VNNI | **16个** | **8次** | 4 |

**核心瓶颈**：vwmacc.vx 是 widening MAC（int8→int16），无法直接产出 int32。每轮只推进 1 个 k，K 维无法并行。

**周期估算**：RVV外积 128-160周期 → vsegdot/x86 VNNI 24-30周期，加速约 **5-6×**。

**YOLO整体推理加速**：

```
计算关系：
  周期减少 X% = (原周期 - 新周期) / 原周期
  加速比 = 1 / (1 - 周期减少比)

YOLO推理热点分布：
  Conv/GEMM（INT8）：35% 热点

收益层级计算：
  1. INT8 GEMM函数收益：周期从256→72
     周期减少 = (256-72)/256 = 72%
     加速比 = 256/72 = 3.56

  2. 整体收益计算（Amdahl定律）：
     p = 0.35（热点占比）, s = 3.56（加速比）
     整体加速比 = 1 / (1 - p + p/s)
                = 1 / (1 - 0.35 + 0.35/3.56)
                = 1 / (0.65 + 0.098)
                = 1 / 0.748 = 1.34
     整体加速 ≈ 34%

     考虑混合精度开销（调整因子0.6-0.8）：
              ≈ 20-27%整体加速
```

### 3.3 方案B：vdot_lane.vx + vsegdot.vv（llama.cpp来源）

**vdot_lane.vx 定义**（来自ARM NEON `vdotq_laneq_s32`）：

```
vdot_lane.vx vd, vs2, vs1, lane_idx
  语义：对于每个i (0 <= i < VL/4)：
    ; 从vs1选择lane_idx对应的4个int8元素
    vd.w[i] += vs2.b[4i]×vs1.b[lane_base] + ... + vs2.b[4i+3]×vs1.b[lane_base+3]
```

**vsegdot.vv 定义**（分段规约点积）：

```
vsegdot.vv vd, vs1, vs2, seg=4
  输入：
    vs1 = vint8m1_t [a0 a1 a2 a3 | a4 a5 a6 a7 | ...]
    vs2 = vint8m1_t [b0 b1 b2 b3 | b4 b5 b6 b7 | ...]
  输出：
    vd = vint32mf2_t（元素数为VL/4）
    vd[0] += a0*b0 + a1*b1 + a2*b2 + a3*b3
    vd[1] += a4*b4 + a5*b5 + a6*b6 + b7*b7
```

**llama.cpp收益量化**（基于指令延迟表）：

| 操作 | 当前RVV | 延迟 | 优化方案 | 延迟 | 减少 |
|------|---------|------|---------|------|------|
| 加载b向量 | vle8.v | 3 | vle8.v | 3 | 不变 |
| Scalar×Vector wid.乘 | vwmul.vx | **9** | — | — | — |
| Widening累加 | vwadd.wv | **4** | vdot_lane.vx | **~7** | **13→7** |
| **每迭代总计** | — | **16** | — | **~10** | **~35%**（K-loop级） |

**llama.cpp整体收益**：

```
收益层级计算：
  1. K-loop级收益：周期从16→10
     周期减少 = (16-10)/16 = 37.5%
     加速比 = 16/10 = 1.60

  2. 函数级收益：K-loop占函数约85%
     函数周期减少 = 85% × 37.5% ≈ 32%
     函数加速比 = 1/(1-0.32) = 1.47

  3. 整体收益计算（Amdahl定律）：
     gemv_q4_0 占比约40%，加速比 1.47
     gemv_q8_0 占比约23%，加速比 1.47
     其他部分占比约37%

     新时间 = 37s + 40s/1.47 + 23s/1.47
            = 37s + 27s + 16s = 80s
     整体加速 = (100-80)/100 = 20%

     考虑实际场景差异（范围估计）：
              ≈ 15-25%整体加速
```

### 3.4 Zvdot4a8i 扩展现状分析

**已存在的分段点积指令**（需`zvdot4a8i`扩展）：

```c
vint32m1_t __riscv_vdot4a_vv_i32m1(vint32m1_t vd, vuint32m1_t vs2, vuint32m1_t vs1, size_t vl);
```

**与理想方案的差距**：

| 对比项 | Zvdot4a8i（现状） | vsegdot.vv（提议） |
|--------|-------------------|-------------------|
| 输入格式 | `vuint32`（打包32位） | 直接`vint8`向量 |
| Signed×Signed | 仅 unsigned variants | 支持 signed×signed |
| 数据预处理 | 需4个int8打包成1个32位 | 无需预处理 |
| 与现有框架兼容 | 需重写CopyPackB逻辑 | 可直接复用ARM模式 |

**结论**：Zvdot4a8i 功能接近但格式不兼容，需提出直接int8输入的新指令方案。

---

## 四、FP32 路径方案详解

### 4.1 方案A：vmulacc.vv（YOLO来源）

**指令定义**：

```
vmulacc.vv vd, vs1, vs2
  功能：vd[4×4矩阵] += vs1[0..3] × vs2[0..3]^T（外积）

  等效操作：16个乘累加
    vd[0,0] += vs1[0] × vs2[0]
    vd[0,1] += vs1[0] × vs2[1]
    ...
    vd[3,3] += vs1[3] × vs2[3]
```

**YOLO收益量化**（FP32 SGEMM，处理2行×16列，K=4步）：

| 实现方式 | 指令数 | 周期数 |
|---------|--------|--------|
| RVV vfmacc×16 | 16条 | 16×5 = 80周期 |
| vmulacc×4 | 4条 | 4×12(预估) = 48周期 |

**收益来源**：
- 指令数减少：75%（函数级）
- B加载共享：4行A共享同一次B加载

**YOLO整体收益**：

```
计算关系：
  周期减少 X% → 加速比 = 1/(1-X%)

收益层级计算：
  1. FP32 GEMM函数收益：周期从80→48
     周期减少 = (80-48)/80 = 40%
     加速比 = 80/48 = 1.67

  2. 整体收益计算（Amdahl定律）：
     FP32 GEMM占YOLO热点约35%

     p = 0.35（热点占比）, s = 1.67（加速比）
     整体加速比 = 1 / (1 - p + p/s)
                = 1 / (0.65 + 0.35/1.67)
                = 1 / (0.65 + 0.21)
                = 1 / 0.86 = 1.16
     整体加速 ≈ 16%

     考虑混合精度开销（调整因子0.6-0.8）：
              ≈ 10-13%整体加速
```

### 4.2 方案B：vfmacc.vv_lane（通用）

**指令定义**：

```
vfmacc.vv_lane vd, vs1, vs2, imm[5-bit]
  功能：vd[i] += vs2[i] × vs1[imm]  (i = 0..VL-1)
        广播vs1的第imm个元素，与vs2逐元素乘加
```

**来源平台**：ARM NEON `fmla v.s[lane]` / `vmlaq_lane_f32`

**收益量化（基于实际RVV SGEMM实现 + 指令延迟表）**：

参考 `applications/onnxrt/rvv-patches/sgemm-kernel-vl16/rvv_sgemm_kernel_vl16.inl` 中 VL=16 内核实现，以 K=2 展开、ProcessTwoRows 模式为例：

**关键指令延迟（来自 `docs/reference/cx/instruction-constraints-and-latency.md`）**：

| 指令 | 延迟（周期） |
|------|-------------|
| `flw`（标量 float load） | 4 |
| `vle32.v`（向量 load） | 3 |
| `vfmacc.vf`（标量广播 FMA） | 5 |

**当前RVV实现周期分析**：

```cpp
// 每次 K=2 迭代（ProcessTwoRows 模式）
float a0_r0 = a[0];        // flw: 4 周期
float a1_r0 = a[1];        // flw: 4 周期
float a0_r1 = a[lda];      // flw: 4 周期
float a1_r1 = a[lda + 1];  // flw: 4 周期

v_b0 = vle32(b);           // vle32.v: 3 周期
vfmacc(v_acc_r0, a0_r0, v_b0);  // vfmacc.vf: 5 周期
vfmacc(v_acc_r1, a0_r1, v_b0);  // vfmacc.vf: 5 周期

v_b1 = vle32(b + 16);      // vle32.v: 3 周期
vfmacc(v_acc_r0, a1_r0, v_b1);  // vfmacc.vf: 5 周期
vfmacc(v_acc_r1, a1_r1, v_b1);  // vfmacc.vf: 5 周期
```

| 操作 | 指令数 | 单条延迟 | 总周期 |
|------|--------|----------|--------|
| 加载 A（标量） | 4 × `flw` | 4 | **16** |
| 加载 B（向量） | 2 × `vle32.v` | 3 | **6** |
| FMA（标量广播） | 4 × `vfmacc.vf` | 5 | **20** |
| **总计** | 10 条 | | **42 周期** |

**扩展后周期分析（Lane-indexed FMA）**：

```asm
vle32.v v_a, (a)                          ; vle32.v: 3 周期（1次向量加载4个A元素）
vle32.v v_b0, (b)                         ; vle32.v: 3 周期
vfmacc.vv_lane v_acc_r0, v_a, v_b0, 0     ; 预估: 5 周期
vfmacc.vv_lane v_acc_r1, v_a, v_b0, 2     ; 预估: 5 周期
vle32.v v_b1, (b+16)                      ; vle32.v: 3 周期
vfmacc.vv_lane v_acc_r0, v_a, v_b1, 1     ; 预估: 5 周期
vfmacc.vv_lane v_acc_r1, v_a, v_b1, 3     ; 预估: 5 周期
```

| 操作 | 指令数 | 单条延迟 | 总周期 |
|------|--------|----------|--------|
| 加载 A（向量） | 1 × `vle32.v` | 3 | **3** |
| 加载 B（向量） | 2 × `vle32.v` | 3 | **6** |
| Lane-FMA | 4 × `vfmacc.vv_lane` | 5（预估） | **20** |
| **总计** | 7 条 | | **29 周期** |

**周期收益对比**（K-loop级）：

| 模式 | 当前周期 | 扩展后周期 | 周期减少（K-loop级） | 指令数减少 |
|------|----------|------------|---------------------|-----------|
| ProcessTwoRows (K=2) | 42 | 29 | **31%** | 30% |
| ProcessOneRow (K=2) | 36 | 19 | **47%** | 40% |
| ProcessOneRow (K=4) | 68 | 27 | **60%** | 44% |

**YOLO整体收益**：

```
计算关系：
  周期减少 X% → 加速比 = 1/(1-X%)
  整体加速比 = 1 / (1 - p + p/s)

收益层级计算：
  1. K-loop级收益：周期减少 31-60%
     加速比范围 = 1.45-2.52

  2. 函数级收益：K-loop占GEMM函数约80%
     函数加速比 = 1/(0.2 + 0.8/s_kloop)
     当s_kloop=1.45：函数加速比 = 1/0.75 = 1.33
     当s_kloop=2.52：函数加速比 = 1/0.52 = 1.93

  3. 整体收益（Amdahl定律）：
     FP32 GEMM占YOLO热点约35%

     当函数加速比=1.33：
     整体加速比 = 1 / (0.65 + 0.35/1.33) = 1/0.91 = 1.10，整体加速 ≈ 10%

     当函数加速比=1.93：
     整体加速比 = 1 / (0.65 + 0.35/1.93) = 1/0.83 = 1.20，整体加速 ≈ 20%

     考虑混合精度开销（调整因子0.6-0.8）：
              ≈ 6-16%整体加速
```

**收益来源解析**：

1. **消除标量加载开销**：4 次 `flw`（4×4=16周期）→ 1 次 `vle32.v`（3周期），节省 **13 周期**
2. **指令数减少**：10 条 → 7 条，减少 **30%**

**与ARM NEON对比**：

| 平台 | 4个K步指令序列 | 总指令数 |
|------|---------------|---------|
| NEON | `ldr q0` + 4×`fmla ... v.s[lane]` | **5 条** |
| RVV现状 | 4×`flw` + 4×`vfmacc.vf` | **8 条** |
| RVV扩展 | `vle32.v` + 4×`vfmacc.vv_lane` | **5 条** |

扩展后 RVV 与 NEON 指令数相当，达到同等效率。

**llama.cpp收益**（作为辅助优化）：
- gemv Scale处理中可替代 `vfwmul.vf` + `vfmacc.vv` 序列
- 整体收益约 **5-8%**

---

## 五、数据重排方案

### 5.1 vunzip/vzip.vv（YOLO来源）

**指令定义**：

```
vunzip.vv vd_even, vd_odd, vs2
  功能：将vs2奇偶元素分离
  输出：vd_even = [e0, e2, e4, ...], vd_odd = [e1, e3, e5, ...]

vzip.vv vd, vs1_even, vs2_odd
  功能：将两向量交错合并
  输出：vd = [e0, o0, e1, o1, ...]
```

**收益量化**（矩阵4×4转置）：

| 实现方式 | 指令数 | 函数级收益 |
|---------|--------|-----------|
| RVV vrgather | 6-8条（含索引向量准备） | 基准 |
| vunzip/vzip | 4条 | **25-40%**（转置函数） |

**应用场景及整体收益**：
- 矩阵转置：**25-40%**（函数级）
- INT8 PackB预处理：**100-300%**（函数级）
- 对YOLO整体：**边际收益**（非热点，占比<5%）

### 5.2 vwmaccwev/wod.vv（llama.cpp来源）

**指令定义**：

```
vwmaccwev.vv vd, vs2, vs1  ; even positions
  vd.h[j] += vs2.b[2j] × vs1.b[2j]

vwmaccwod.vv vd, vs2, vs1  ; odd positions
  vd.h[j] += vs2.b[2j+1] × vs1.b[2j+1]
```

**收益**：
- 减少依赖链深度（函数级优化）
- llama.cpp整体收益：**3-5%**

---

## 六、整体收益汇总

### 6.1 按应用分列（整体收益）

| 应用 | 扩展组合 | 整体收益 |
|------|---------|----------|
| **llama.cpp** | vdot_lane.vx（gemv_q4+q8） | **15-25%** |
| **llama.cpp** | vsegdot.vv（gemv系列） | **15-25%**（同上，等效方案） |
| **llama.cpp** | vfmacc.vv_lane（辅助） | **5-8%** |
| **llama.cpp** | vwmaccwev/wod.vv | **3-5%** |
| **YOLO** | vsegdot.vv（INT8路径） | **20-26%** |
| **YOLO** | vmulacc + lane-FMA（FP32路径） | **10-16%** |
| **YOLO** | 全套（含vzip） | **25-35%**（组合估算） |

### 6.2 跨应用累计收益估算

若同时应用于两个应用（假设相同硬件平台）：

| 场景 | 整体收益 |
|------|---------|
| llama.cpp推理优化 | **15-25%** |
| YOLO推理优化 | **25-35%** |
| **综合提升** | 两个应用均受益于INT8分段点积指令 |

---

## 七、与主流架构对比

### 7.1 扩展后性能定位（相对吞吐量）

| 指标 | x86 VNNI | ARM SDOT | RVV现状 | RVV扩展后 |
|------|---------|---------|---------|----------|
| INT8 GEMM吞吐 | 100%（基准） | 80-100% | **30-50%** | **85-100%** |
| FP32 GEMM吞吐 | 100%（基准） | 90% | 60% | **85-100%** |
| 矩阵转置效率 | 高 | 高 | 低 | 中 |
| Lane-indexed操作 | 有 | 有 | **无** | 有 |

### 7.2 关键差距填补

| 差距类型 | RVV现状 | 提议方案 | 填补效果 |
|---------|---------|---------|---------|
| 分段规约（INT8） | 仅vredsum（单输出） | vsegdot.vv | **达到VNNI水平** |
| Lane-indexed | 仅.vx标量广播 | vfmacc_lane/vdot_lane | **消除广播开销** |
| 矩阵外积 | 需16条vfmacc | vmulacc（1条） | **减少75%指令** |

---

## 八、当前RVV关键指令延迟

基于 `docs/reference/cx/instruction-constraints-and-latency.md`：

| 指令 | 延迟 | 说明 |
|------|------|------|
| vle8.v | 3 | 加载 |
| vwmul.vx (SEW=8) | **4+5 = 9** | 标量广播开销 |
| vwmul.vv (SEW=8) | **4** | 向量乘法 |
| vwadd.wv | 4 | Widening累加 |
| vwmacc.vx | 4+5 | Widening MAC（scalar） |
| vwmacc.vv | 5 | Widening MAC（vector） |
| vfmacc.vv | 5 | FP32 FMA |
| vredsum.vs | 4 | 规约（单输出） |
