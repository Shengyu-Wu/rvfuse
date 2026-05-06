# BBV Hotspot Analysis Guide

This file contains detailed best practices for analyzing BBV output with `analyze_bbv.py`.
For usage instructions, see SKILL.md.

## Table of Contents

1. [Correct Sysroot Path](#correct-sysroot-path)
2. [PIE Executable Address Resolution](#pie-executable-address-resolution)
3. [Application Library Matching Priority](#application-library-matching-priority)
4. [Verifying Analysis Results](#verifying-analysis-results)
5. [Stripped Binary Handling](#stripped-binary-handling)

## Correct Sysroot Path

**Critical**: The sysroot path must contain all shared libraries used by the profiled binary.
Using an incorrect sysroot leads to misidentification of hotspots.

```bash
# Example: llama.cpp has its own sysroot with libllama.so, libggml*.so
python3 tools/analyze_bbv.py \
  --bbv output/llama.bbv.0.bb \
  --elf output/llama.cpp/bin/llama-cli \
  --sysroot output/llama.cpp/sysroot   # NOT output/sysroot!
```

**Verification**: Check that application-specific libraries exist in the sysroot:
```bash
# For llama.cpp
ls output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama*.so
ls output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libggml*.so

# For ONNX Runtime
ls output/sysroot/lib/riscv64-linux-gnu/libonnx*.so
```

## PIE Executable Address Resolution

Modern binaries are often PIE (Position Independent Executable). Runtime addresses
differ from static file offsets, requiring base address detection.

The `analyze_bbv.py` script handles this automatically:
- Detects PIE executables via `file` command output
- Analyzes address distribution to estimate runtime base
- Converts runtime addresses to file offsets for addr2line

**Manual verification** (if needed):
```bash
# Check if binary is PIE
file output/llama.cpp/bin/llama-cli
# Should show: "DYN ... PIE executable"

# Estimated PIE base appears in stderr during analysis
python3 tools/analyze_bbv.py --bbv ... 2>&1 | grep -i "pie"
```

## Application Library Matching Priority

The analysis prioritizes application-specific libraries (llama, ggml, onnx, ort) over
system libraries (libc, libcrypto). This prevents hotspots being incorrectly attributed
to system libraries.

**Common misidentification patterns** (before fix):
- libcrypto.so showing 80%+ of execution (incorrect)
- libgnutls.so showing most hotspots (incorrect)

**Expected patterns** (after proper analysis):
- libllama.so showing majority of inference hotspots
- libggml-cpu.so showing quantization/computation hotspots
- libonnxruntime.so showing ML inference hotspots

## Verifying Analysis Results

### Library Distribution Check

```bash
python3 -c "
import json
with open('output/hotspot.json') as f:
    data = json.load(f)
libs = {}
for b in data['blocks']:
    loc = b['location']
    if '[' in loc:
        lib = loc.split('[')[1].split(']')[0]
        libs[lib] = libs.get(lib, 0) + b['count']
total = sum(libs.values())
for lib, cnt in sorted(libs.items(), key=lambda x: -x[1])[:5]:
    print(f'{lib}: {cnt/total*100:.2f}%')
"
```

**Expected**: Application libraries (libllama, libggml, libonnx) should dominate,
not system libraries.

### Symbol Resolution Check

Hotspots should show meaningful function names, not `??`:

```bash
head -20 output/hotspot-report.txt | grep -E "^\s*[0-9]+"
```

**Good result**: `[libggml-cpu.so] ggml_quantize_mat_q8_0_4x4`
**Bad result**: `[libcrypto.so.3] ?? (??:0)` (wrong library + no symbol)

### Address-to-Library Mapping

For suspicious addresses, manually verify:

```bash
addr=0x7f290b083500
base=0x7f290b04f000
offset=$((addr - base))
addr2line -f -e output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libggml-cpu.so.0.9.11 $offset
```

## Stripped Binary Handling

Most RISC-V libraries are stripped (no debug symbols). The analysis still works because:
- Dynamic symbol table (`nm -D`) provides function names
- Library matching uses LOAD segment address ranges
- Hotspots are attributed to correct libraries even without file/line info

```bash
file output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama.so.0.0.1
# Shows: "stripped" - but nm -D still shows exported symbols

nm -D output/llama.cpp/sysroot/lib/riscv64-linux-gnu/libllama.so.0.0.1 | head
```

## Analysis Algorithm Details

The `analyze_bbv.py` script uses these techniques:

1. **PIE base detection**: Analyzes low-address cluster (0x55* region) to estimate base
2. **Application library priority**: Processes llama/ggml/onnx libs before libc/libcrypto
3. **Execution count weighting**: Weights candidate bases by total execution count
4. **Symbol validation bonus**: Uses addr2line to verify symbols, giving 50% bonus