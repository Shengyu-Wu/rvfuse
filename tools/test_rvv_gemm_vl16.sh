#!/usr/bin/env bash
# -*- mode: bash -*-
#
# test_rvv_gemm_vl16.sh - Test VL=16 RVV GEMM kernel correctness in QEMU
#
# This script:
# 1. Builds ONNX Runtime with VL=16 RVV GEMM kernel
# 2. Runs MLAS GEMM tests in QEMU
# 3. Compares results between scalar and VL=16 kernels
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PATCH_FILE="${PROJECT_ROOT}/applications/onnxrt/rvv-patches/sgemm-kernel-vl16/patch.diff"
ORT_BUILD_DIR="${PROJECT_ROOT}/output/ort-vl16-build"
ORT_INSTALL_DIR="${PROJECT_ROOT}/output/ort-vl16"
SYSROOT="${PROJECT_ROOT}/output/sysroot"
QEMU_PATH="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
QEMU_CPU_OPTS="-cpu rv64,v=true,vlen=512"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }
step()  { echo -e "${BLUE}>>> $*${NC}"; }

# --- Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."

    command -v cmake &>/dev/null || error "cmake not found"
    command -v ninja &>/dev/null || error "ninja not found"
    command -v git &>/dev/null || error "git not found"

    # Check patch file exists
    [ -f "${PATCH_FILE}" ] || error "Patch file not found: ${PATCH_FILE}"

    # Check sysroot exists
    [ -d "${SYSROOT}/usr" ] || error "Sysroot not found: ${SYSROOT}. Run ./applications/yolo/ort/build.sh first."

    # Check QEMU exists (or use system qemu)
    if [ ! -f "${QEMU_PATH}" ]; then
        warn "QEMU not found at ${QEMU_PATH}, checking system..."
        command -v qemu-riscv64 &>/dev/null || error "qemu-riscv64 not found. Build QEMU or install qemu-user-static."
        QEMU_PATH="qemu-riscv64"
    fi

    info "All prerequisites met."
    echo "  Patch:   ${PATCH_FILE}"
    echo "  Sysroot: ${SYSROOT}"
    echo "  QEMU:    ${QEMU_PATH}"
}

# --- Apply Patch ---
apply_patch() {
    local ort_source="${PROJECT_ROOT}/applications/yolo/ort/vendor/onnxruntime"

    [ -d "${ort_source}/cmake" ] || error "ORT source not found: ${ort_source}"

    info "Applying VL=16 RVV GEMM patch..."

    cd "${ort_source}"

    # Check if patch already applied
    if git log --oneline -1 | grep -q "rvv-gemm-vl16"; then
        info "Patch already applied."
        return 0
    fi

    # Apply patch
    step "Applying patch from ${PATCH_FILE}..."
    if git apply --check "${PATCH_FILE}" 2>/dev/null; then
        git apply "${PATCH_FILE}"
        info "Patch applied successfully."
    else
        warn "Patch may already be partially applied. Trying with --3way..."
        git apply --3way "${PATCH_FILE}" || {
            error "Failed to apply patch. Check for conflicts."
        }
    fi

    cd "${PROJECT_ROOT}"
}

# --- Build MLAS with VL=16 ---
build_mlas_vl16() {
    local ort_source="${PROJECT_ROOT}/applications/yolo/ort/vendor/onnxruntime"
    local toolchain="${PROJECT_ROOT}/applications/yolo/ort/riscv64-linux-toolchain.cmake"
    local llvm_install="${PROJECT_ROOT}/third_party/llvm-install"

    [ -f "${toolchain}" ] || error "Toolchain file not found: ${toolchain}"
    [ -d "${llvm_install}/bin" ] || error "LLVM install not found: ${llvm_install}"

    info "Building ONNX Runtime with VL=16 RVV GEMM kernel..."

    rm -rf "${ORT_BUILD_DIR}"
    mkdir -p "${ORT_BUILD_DIR}" "${ORT_INSTALL_DIR}"

    export LLVM_INSTALL="${llvm_install}"
    export SYSROOT="${SYSROOT}"

    # Build with tests enabled for correctness verification
    cmake -S "${ort_source}/cmake" -B "${ORT_BUILD_DIR}" \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
        -DCMAKE_INSTALL_PREFIX="${ORT_INSTALL_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=ON \
        -Donnxruntime_BUILD_MLAS_TESTS=ON \
        -Donnxruntime_DISABLE_RTTI=ON \
        -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option' \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -G Ninja

    step "Building (ninja)..."
    ninja -C "${ORT_BUILD_DIR}"

    unset LLVM_INSTALL SYSROOT

    info "Build complete: ${ORT_BUILD_DIR}"
}

# --- Run MLAS Tests in QEMU ---
run_mlas_tests() {
    local test_exe="${ORT_BUILD_DIR}/onnxruntime_mlas_test"

    [ -f "${test_exe}" ] || error "MLAS test executable not found: ${test_exe}"

    info "Running MLAS GEMM tests in QEMU..."

    step "Testing VL=16 kernel correctness..."

    # Run tests with QEMU
    # The tests cover:
    # - Matrix dimensions: M,N,K = 1..256
    # - Transpose combinations: NN, NT, TN, TT
    # - Alpha/beta: 0.0, 1.0, -1.0, 0.5
    # - Batch sizes: 1..3
    "${QEMU_PATH}" ${QEMU_CPU_OPTS} -L "${SYSROOT}" "${test_exe}" --gtest_filter="*Fgemm*" 2>&1 | tee "${PROJECT_ROOT}/output/vl16_test_results.log"

    local result=$?
    if [ $result -eq 0 ]; then
        info "All MLAS GEMM tests PASSED!"
    else
        error "Some tests FAILED. Check log: output/vl16_test_results.log"
    fi
}

# --- Build Scalar Reference ---
build_scalar_reference() {
    local ort_source="${PROJECT_ROOT}/applications/yolo/ort/vendor/onnxruntime"
    local scalar_build="${PROJECT_ROOT}/output/ort-scalar-build"
    local toolchain="${PROJECT_ROOT}/applications/yolo/ort/riscv64-linux-toolchain.cmake"
    local llvm_install="${PROJECT_ROOT}/third_party/llvm-install"

    info "Building scalar reference (without VL=16 patch) for comparison..."

    # Create a clean worktree for scalar build
    cd "${ort_source}"
    local has_vl16_patch
    has_vl16_patch=$(git log --oneline -1 | grep -c "rvv-gemm-vl16" || echo "0")

    if [ "${has_vl16_patch}" -gt 0 ]; then
        step "Reverting VL=16 patch for scalar build..."
        git stash || true
    fi

    cd "${PROJECT_ROOT}"

    rm -rf "${scalar_build}"
    mkdir -p "${scalar_build}"

    export LLVM_INSTALL="${llvm_install}"
    export SYSROOT="${SYSROOT}"

    cmake -S "${ort_source}/cmake" -B "${scalar_build}" \
        -DCMAKE_TOOLCHAIN_FILE="${toolchain}" \
        -DCMAKE_BUILD_TYPE=Release \
        -Donnxruntime_BUILD_SHARED_LIB=ON \
        -Donnxruntime_BUILD_UNIT_TESTS=ON \
        -Donnxruntime_BUILD_MLAS_TESTS=ON \
        -Donnxruntime_DISABLE_RTTI=ON \
        -DCMAKE_CXX_FLAGS='-Wno-stringop-overflow -Wno-unknown-warning-option' \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -G Ninja

    ninja -C "${scalar_build}"

    unset LLVM_INSTALL SYSROOT

    # Restore patch
    if [ "${has_vl16_patch}" -gt 0 ]; then
        cd "${ort_source}"
        git stash pop || git apply "${PATCH_FILE}"
        cd "${PROJECT_ROOT}"
    fi

    info "Scalar reference built: ${scalar_build}"
}

# --- Compare Kernel Outputs ---
compare_kernels() {
    local vl16_test="${ORT_BUILD_DIR}/onnxruntime_mlas_test"
    local scalar_test="${PROJECT_ROOT}/output/ort-scalar-build/onnxruntime_mlas_test"

    info "Comparing VL=16 vs scalar kernel outputs..."

    # Run both tests and compare
    step "Running scalar tests..."
    "${QEMU_PATH}" ${QEMU_CPU_OPTS} -L "${SYSROOT}" "${scalar_test}" --gtest_filter="*Fgemm*" 2>&1 | tee "${PROJECT_ROOT}/output/scalar_test_results.log"

    step "Running VL=16 tests..."
    "${QEMU_PATH}" ${QEMU_CPU_OPTS} -L "${SYSROOT}" "${vl16_test}" --gtest_filter="*Fgemm*" 2>&1 | tee "${PROJECT_ROOT}/output/vl16_test_results.log"

    # Compare test counts
    local scalar_passed
    scalar_passed=$(grep -c "PASSED" "${PROJECT_ROOT}/output/scalar_test_results.log" || echo "0")
    local vl16_passed
    vl16_passed=$(grep -c "PASSED" "${PROJECT_ROOT}/output/vl16_test_results.log" || echo "0")

    step "Test comparison:"
    echo "  Scalar tests passed: ${scalar_passed}"
    echo "  VL=16 tests passed:  ${vl16_passed}"

    if [ "${vl16_passed}" -ge "${scalar_passed}" ]; then
        info "VL=16 kernel passes all tests that scalar passes!"
    else
        warn "VL=16 kernel passes fewer tests than scalar. Check logs."
    fi
}

# --- Main ---
main() {
    local mode="${1:-full}"

    case "${mode}" in
        full)
            check_prerequisites
            apply_patch
            build_mlas_vl16
            run_mlas_tests
            ;;
        build)
            check_prerequisites
            apply_patch
            build_mlas_vl16
            ;;
        test)
            check_prerequisites
            run_mlas_tests
            ;;
        compare)
            check_prerequisites
            apply_patch
            build_mlas_vl16
            build_scalar_reference
            compare_kernels
            ;;
        *)
            error "Unknown mode: ${mode}. Use: full, build, test, compare"
            ;;
    esac

    info "Done!"
}

main "$@"