#!/usr/bin/env bash
# ResNet50 ONNX Runtime inference test demo
# Tests the generic_ort_runner with ResNet50 model on RISC-V (via QEMU or Banana Pi)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-ort"
MODEL_DIR="${PROJECT_ROOT}/output/models"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
MODE="${1:-qemu}"
ITERATIONS="${2:-10}"

# --- Check prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."

    # Runner
    [ -f "${OUTPUT_DIR}/generic_ort_runner" ] || error "generic_ort_runner not found. Run build first."
    file "${OUTPUT_DIR}/generic_ort_runner"

    # ORT library
    [ -f "${OUTPUT_DIR}/lib/libonnxruntime.so" ] || error "libonnxruntime.so not found."
    ls -lh "${OUTPUT_DIR}/lib/libonnxruntime.so.1.24.4"

    # Sysroot (for QEMU)
    [ -d "${OUTPUT_DIR}/sysroot" ] || error "Sysroot not found."

    # Model
    [ -f "${MODEL_DIR}/resnet50.onnx" ] || {
        warn "ResNet50 model not found. Downloading..."
        wget -q -O "${MODEL_DIR}/resnet50.onnx" \
            "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
    }
    ls -lh "${MODEL_DIR}/resnet50.onnx"

    info "All prerequisites met."
}

# --- QEMU local test ---
run_qemu_test() {
    info "Running ResNet50 inference via QEMU..."

    # QEMU with sysroot
    QEMU_CMD="qemu-riscv64 -L ${OUTPUT_DIR}/sysroot"

    # Set library path
    export QEMU_LD_PREFIX="${OUTPUT_DIR}/sysroot"
    export LD_LIBRARY_PATH="${OUTPUT_DIR}/lib:${OUTPUT_DIR}/sysroot/usr/lib/riscv64-linux-gnu"

    # Run inference
    ${QEMU_CMD} "${OUTPUT_DIR}/generic_ort_runner" "${MODEL_DIR}/resnet50.onnx" "${ITERATIONS}"

    info "QEMU test completed."
}

# --- Banana Pi perf profiling ---
# Use the perf-profiling skill instead of a dedicated script.
# Example: ask Claude to run perf profiling with:
#   "在香蕉派上对 ResNet50 做 perf profiling，IP=${RVFUSE_HOST}"
run_perf_profile() {
    info "Perf profiling is now handled by the perf-profiling skill."
    info "Ask Claude: '在香蕉派 ${RVFUSE_HOST:-<IP>} 上对 ResNet50 做 perf profiling，${ITERATIONS} 次迭代'"
    info "The skill handles: upload → chroot → perf stat/record/report → download → summary"
}

# --- Main ---
check_prerequisites

echo ""
echo "Available test modes:"
echo "  qemu     - Local QEMU simulation (default)"
echo "  perf     - Banana Pi perf profiling (via perf-profiling skill)"
echo ""
echo "Running mode: ${MODE}"
echo "Iterations: ${ITERATIONS}"
echo ""

case "${MODE}" in
    qemu)   run_qemu_test ;;
    perf)   run_perf_profile ;;
    *)      error "Unknown mode: ${MODE}" ;;
esac

echo ""
echo "Test completed successfully!"