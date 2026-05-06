---
name: perf-profiling
description: |
  Profile RISC-V inference workloads on real hardware (Banana Pi) using Linux perf.
  Handles the full workflow: model download, artifact upload via scp, perf profiling
  via ssh, result download, and summary generation. Supports ONNX Runtime
  (generic_ort_runner) and llama.cpp frameworks. Auto-detects framework from model
  file extension (.onnx / .gguf). No external scripts required — all steps executed
  directly via ssh/scp commands.
  Use this skill when the user mentions: perf profiling, hardware profiling,
  香蕉派 profiling, real hardware hotspot, instruction fusion hotspot discovery,
  多模型 profiling, perf stat, perf record, perf annotate, remote profiling,
  远程 profiling, or replacing QEMU BBV with hardware profiling.
---

# Perf Profiling on RISC-V Hardware

Profile inference workloads on real RISC-V hardware (Banana Pi) using Linux `perf`.
Execute the full workflow directly via `ssh`/`scp` — no Python scripts needed.

Unlike QEMU BBV profiling, hardware perf captures real cycle counts, cache behavior,
and branch prediction effects.

## Execution Phases

Follow these phases in order. Adapt commands based on the user's request (framework,
models, iterations, etc.).

### Phase 1: Model Resolution

Resolve each model the user specifies. Resolution order:
1. Exact local file path — use as-is if it exists
2. `output/models/<basename>` — check if already downloaded
3. Registry short name — download from URL below
4. Direct URL — download with `wget`

**Model Registry:**

| Short Name | Framework | Filename | Download URL |
|------------|-----------|----------|-------------|
| `resnet50` | ORT | `resnet50.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx` |
| `mobilenetv2` | ORT | `mobilenetv2.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx` |
| `squeezenet` | ORT | `squeezenet.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.1-7.onnx` |
| `shufflenet` | ORT | `shufflenet.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/shufflenet/model/shufflenet-9.onnx` |
| `vgg16` | ORT | `vgg16.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg16-7.onnx` |
| `densenet121` | ORT | `densenet121.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/densenet-121/model/densenet-121.onnx` |
| `inception` | ORT | `inception_v1.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.onnx` |
| `efficientnet-lite4` | ORT | `efficientnet-lite4.onnx` | `https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx` |
| `qwen-0.5b-q4_0` | llama.cpp | `Qwen2.5-0.5B-Instruct-Q4_0.gguf` | `https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_0.gguf` |
| `qwen-1.5b-q4_0` | llama.cpp | `Qwen2.5-1.5B-Instruct-Q4_0.gguf` | `https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_0.gguf` |

If download needed:
```bash
mkdir -p output/models
wget -q -O output/models/<filename> "<url>"
```

### Phase 2: Detect Framework and Mode

**Framework detection** from model file extension:
- `.onnx` / `.ort` → `ort` (ONNX Runtime)
- `.gguf` → `llama` (llama.cpp)

**Mode detection** — check if `rootfs.tar.gz` exists alongside the runner:
```bash
# ORT default: output/cross-ort/rootfs.tar.gz
# llama.cpp: typically no rootfs, use --libs + --sysroot
```

If `rootfs.tar.gz` exists → use **chroot mode** (recommended for ORT).
Otherwise → use **direct mode** with `LD_LIBRARY_PATH`.

### Phase 3: Upload

Collect connection info from user (or use defaults):
- `HOST`: Banana Pi IP (required)
- `USER`: SSH username (default: `root`)
- `REMOTE_DIR`: remote working directory (default: `/root`)
- `RUNNER`: local path to inference binary

Use `sshpass` for non-interactive password auth, or `ssh-copy-id` for key-based.

```bash
SSH_CMD="sshpass -p '<password>' ssh -o StrictHostKeyChecking=no <user>@<host>"
SCP_CMD="sshpass -p '<password>' scp -o StrictHostKeyChecking=no"
REMOTE=<remote-dir>
```

#### Chroot Mode (ORT recommended)

```bash
# Upload and extract rootfs
$SSH_CMD "mkdir -p $REMOTE"
$SCP_CMD output/cross-ort/rootfs.tar.gz $REMOTE/rootfs.tar.gz
$SSH_CMD "cd $REMOTE && tar xzf rootfs.tar.gz"

# Copy models into rootfs
$SCP_CMD <model_path> $REMOTE/rootfs/<model_filename>
```

#### Direct Mode (llama.cpp or no rootfs)

```bash
$SSH_CMD "mkdir -p $REMOTE/lib"

# Upload runner
$SCP_CMD <runner> $REMOTE/<runner_name>
$SSH_CMD "chmod +x $REMOTE/<runner_name>"

# Upload shared libraries
scp <libs_dir>/*.so* $REMOTE/lib/

# Upload sysroot (if needed)
scp -r <sysroot> $REMOTE/sysroot

# Upload model
$SCP_CMD <model_path> $REMOTE/<model_filename>
```

### Phase 4: Setup and Profile

First, check the remote environment:

```bash
$SSH_CMD "uname -m && perf --version 2>&1"
$SSH_CMD "cat /proc/sys/kernel/perf_event_paranoid"
# If paranoid > 1:
$SSH_CMD "echo 0 > /proc/sys/kernel/perf_event_paranoid"
```

#### Chroot Setup (if using rootfs)

```bash
$SSH_CMD "mkdir -p $REMOTE/rootfs/{proc,dev,sys,tmp}"
$SSH_CMD "mount -t proc proc $REMOTE/rootfs/proc 2>/dev/null || true"
$SSH_CMD "mount -t sysfs sysfs $REMOTE/rootfs/sys 2>/dev/null || true"
$SSH_CMD "mount --bind /dev $REMOTE/rootfs/dev 2>/dev/null || true"
```

#### Build Runner Command

**ORT (chroot mode):**
```
run_cmd = "chroot $REMOTE/rootfs /<runner_name> <model_filename> <iterations>"
```

**ORT (direct mode):**
```
run_cmd = "LD_LIBRARY_PATH=$REMOTE/lib $REMOTE/<runner_name> <model_filename> <iterations>"
```

**llama.cpp:**
```
run_cmd = "LD_LIBRARY_PATH=$REMOTE/lib $REMOTE/llama-cli -m $REMOTE/<model.gguf> <input_args>"
```

#### Run perf (per model)

```bash
WORK=$REMOTE/perf_<model_stem>
$SSH_CMD "mkdir -p $WORK"

# 1. perf stat — global metrics
$SSH_CMD "perf stat -d -o $WORK/perf_stat.txt -- $run_cmd"

# 2. perf record — sampling (cpu-clock for RISC-V, see notes below)
$SSH_CMD "perf record -e cpu-clock -g -F <freq> -o $WORK/perf.data -- $run_cmd"

# 3. Generate reports
SYMFS=""
# If chroot mode: SYMFS="--symfs $REMOTE/rootfs"
$SSH_CMD "perf report --stdio -n --percent-limit 0.5 $SYMFS -i $WORK/perf.data > $WORK/perf_report.txt 2>/dev/null"
$SSH_CMD "perf annotate --stdio $SYMFS -i $WORK/perf.data > $WORK/perf_annotate.txt 2>/dev/null"
```

### Phase 5: Download and Summarize

```bash
# Create local output directory
mkdir -p <outdir>/<model_stem>

# Download results
$SCP_CMD $HOST:$WORK/perf_stat.txt <outdir>/<model_stem>/
$SCP_CMD $HOST:$WORK/perf_report.txt <outdir>/<model_stem>/
$SCP_CMD $HOST:$WORK/perf_annotate.txt <outdir>/<model_stem>/
```

After downloading all models, read the local `perf_stat.txt` and `perf_report.txt`
files and generate `<outdir>/summary.md` with:
- Table of models vs. metrics (cycles, instructions, IPC, cache miss %, top function)
- Top-10 hot functions per model
- Cross-model shared hot functions → priority fusion targets

### Phase 6: Cleanup

```bash
# Remove perf data from remote
$SSH_CMD "rm -rf $REMOTE/perf_<model_stem>"

# If chroot mode, unmount
$SSH_CMD "umount $REMOTE/rootfs/proc 2>/dev/null || true"
$SSH_CMD "umount $REMOTE/rootfs/sys 2>/dev/null || true"
$SSH_CMD "umount $REMOTE/rootfs/dev 2>/dev/null || true"
```

---

## Supported Frameworks

### ONNX Runtime (generic_ort_runner)

Cross-compiled via `applications/yolo/ort/build.sh` (see cross-compile-app skill).
The `generic_ort_runner` reads input shape from the ONNX model itself and generates
random test data — no need to write model-specific test cases.

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| Runner binary | `output/cross-ort/generic_ort_runner` | `<remote-dir>/rootfs/generic_ort_runner` |
| ORT shared lib | `output/cross-ort/lib/libonnxruntime.so*` | inside rootfs.tar.gz |
| Rootfs (chroot) | `output/cross-ort/rootfs.tar.gz` | `<remote-dir>/rootfs.tar.gz` |
| ONNX models | `*.onnx` or `*.ort` files | `<remote-dir>/rootfs/` |

### llama.cpp

Cross-compiled via `applications/llama.cpp/build.sh` (see cross-compile-app skill).

| Artifact | Local Path | Remote Path |
|----------|-----------|-------------|
| CLI binary | `output/llama.cpp/bin/llama-cli` | `<remote-dir>/llama-cli` |
| Core library | `output/llama.cpp/lib/libllama.so*` | `<remote-dir>/lib/` |
| GGML backends | `output/llama.cpp/lib/libggml*.so*` | `<remote-dir>/lib/` |
| Sysroot | `output/llama.cpp/sysroot/` | `<remote-dir>/sysroot/` |
| GGUF models | `*.gguf` files | `<remote-dir>/` |

---

## RISC-V Perf Notes

### Software Events Required

Banana Pi (SpacemiT K1) SBI PMU does not reliably support hardware cycle sampling.
Use `-e cpu-clock` (software event) which always works:

```bash
# This works on Banana Pi:
perf record -e cpu-clock -g -F 999 -- ./runner model.onnx

# This may fail (no samples):
perf record -e cycles -g -F 999 -- ./runner model.onnx
```

### Symbol Resolution

For meaningful function names in `perf report`:
1. Build with `-g` (debug info) and without stripping
2. `generic_ort_runner` and `yolo_inference` binaries have debug info by default
3. **libonnxruntime.so** must NOT be stripped — build script uses `install` (not `install/strip`)
4. Shared libs must be in `LD_LIBRARY_PATH` on the remote board

---

## Profiling Commands (Quick Reference)

### perf stat — Global Metrics

```bash
perf stat -d -- ./generic_ort_runner model.onnx 30
```

| Metric | Meaning | Fusion Relevance |
|--------|---------|------------------|
| `cycles` | Total CPU cycles | Baseline for speedup estimation |
| `instructions` | Total instructions retired | Instruction mix density |
| `IPC` | Instructions per cycle | Compute vs memory bottleneck |
| `cache-misses` | Cache miss rate | Memory-bound vs compute-bound |
| `branch-misses` | Mispredicted branches | Control-flow hotspot |

### perf record — Sampling

```bash
perf record -e cpu-clock -g -F 999 -o perf.data -- ./generic_ort_runner model.onnx 30
```

### perf report — Function Hotspots

```bash
perf report --stdio -n --percent-limit 0.5 -i perf.data
```

### perf annotate — Instruction Hotspots

```bash
perf annotate --stdio -i perf.data
```

Output format:
```
 Percent |      Source code & Disassembly
---------------------------------------------------------
 12.30 :   106b4:   ld      a5,0(a0)
  8.72 :   106b8:   addiw   a5,a5,1
  7.45 :   106bc:   mulw    a5,a5,a4
  5.21 :   106c0:   sw      a5,0(a0)
```

Instructions with >5% are strong fusion candidates.

---

## Hotspot Analysis for Fusion Candidates

### Fusion Patterns to Look For

**1. Load-Compute-Store (Memory-ALU fusion)**
```
 15.2 :   ld      a5,0(a0)
  9.8 :   addw    a5,a5,a4
 12.1 :   sw      a5,0(a0)
```

**2. Multiply-Accumulate Chain (MAC fusion)**
```
  8.5 :   mulw    a5,a3,a4
  7.2 :   addw    a5,a5,a6
```

**3. Address Calculation (Address generation fusion)**
```
  5.1 :   slli    a5,a5,2
  4.3 :   add     a5,a5,a0
  4.8 :   ld      a5,0(a5)
```

### Analysis Checklist

1. **Top-10 functions**: Which consume >80% of cycles?
2. **IPC**: Compute-bound (IPC < 1.0) or memory-bound?
3. **Cross-model overlap**: Shared hot functions = highest priority fusion targets
4. **Cross-framework overlap**: Functions hot in both ORT and llama.cpp = universal candidates

---

## Output Directory Convention

```
output/perf/
├── ort/                       # ONNX Runtime results
│   ├── resnet50/
│   │   ├── perf_stat.txt
│   │   ├── perf_report.txt
│   │   └── perf_annotate.txt
│   ├── mobilenetv2/
│   └── summary.md
└── llama/                     # llama.cpp results
    ├── qwen-0.5b/
    └── summary.md
```

---

## Prerequisites

### Remote Board

```bash
perf --version
sudo sysctl -w kernel.perf_event_paranoid=0
```

### Local Machine

```bash
# sshpass for non-interactive scp/ssh
sudo apt install sshpass
```

### Cross-Compilation

Use the **cross-compile-app** skill to build ONNX Runtime or llama.cpp for RISC-V
before profiling. Key flags for good profiling:
- `-g -fno-omit-frame-pointer` for perf call graph unwinding
- Do NOT strip shared libraries (symbols needed for `perf report`)
- Use `generic_ort_runner` for ORT — it auto-detects model input shape

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Permission denied | perf_event_paranoid too high | `ssh ... "echo 0 > /proc/sys/kernel/perf_event_paranoid"` |
| No symbols (hex addresses) | Binary stripped or no debug info | Build with `-g -fno-omit-frame-pointer`; use `install` not `install/strip` |
| No symbols in chroot mode | perf can't find .so inside rootfs | Use `--symfs <rootfs-path>` in perf report/annotate |
| No samples in perf record | Hardware PMU not supported on SBI | Use `-e cpu-clock` instead of `-e cycles` |
| Too few samples | Low freq or short run | Increase `-F 9999` or iterations to 100+ |
| High overhead | Frequency too high | Use `-F 99` |
| SSH connection fails | Network or auth issue | Test: `sshpass -p '<pwd>' ssh <user>@<host> uname -m` |
| glibc version mismatch | Board glibc older than build sysroot | Use rootfs.tar.gz for chroot isolation |
| chroot fails | No root on board | SSH as root user (default on Banana Pi) |

## Limitations

| Limitation | Mitigation |
|------------|------------|
| SBI PMU events vary by firmware | Check `perf list`; software events always work |
| perf is statistical, not exact | Increase `-F` for more resolution |
| Limited PMU counters | Profile events in separate runs |
| cpu-clock samples at lower rate | Use high freq (999+) and more iterations (30+) |
| chroot requires root | SSH as root user (default on Banana Pi) |
