# YOLO Perf Profiling Data

## Naming Convention

```
<model>_<board>_<soc>_<isa>_<kernel_type>_<YYYYMMDD>_<content>.ext
```

| Field | Example | Description |
|-------|---------|-------------|
| `model` | `yolo11n` | Model name |
| `board` | `bananapi` | Development board |
| `soc` | `k1` | SoC name (SpacemiT K1) |
| `isa` | `rv64gcv` | RISC-V ISA string |
| `kernel_type` | `scalar` | GEMM kernel type: `scalar` = flw/fmadd.s |
| `date` | `20260424` | Profiling date |
| `content` | `analysis` / `perf_stat` / `perf_report` / `perf_annotate` | Content type |

## Content Types

| Suffix | Tool | Content |
|--------|------|---------|
| `perf_stat.txt` | `perf stat -d` | Global metrics: cycles, instructions, IPC, cache, branches |
| `perf_report.txt` | `perf report --stdio` | Function-level hotspots (children + self %) |
| `perf_annotate.txt` | `perf annotate --stdio` | Instruction-level hotspots with source |
| `analysis.md` | manual | Human-readable analysis with fusion candidates |

## Data Files

| File | Date | ISA | Kernel | Description |
|------|------|-----|--------|-------------|
| `yolo11n_bananapi_k1_rv64gcv_scalar_20260424_*` | 2026-04-24 | rv64gcv | scalar | YOLO11n ORT v1.24.4, scalar GEMM kernels |
