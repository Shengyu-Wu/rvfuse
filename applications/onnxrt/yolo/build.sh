#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/cross-ort"
RUNNER_DIR="${SCRIPT_DIR}/runner"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Argument parsing ---
FORCE=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)         FORCE=true; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -j*)             JOBS="${1#-j}"; shift ;;
        *)               error "Unknown argument: $1" ;;
    esac
done

# --- Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang++" ] || error "clang++ not found at ${LLVM_INSTALL}/bin/clang++"
    [ -d "${OUTPUT_DIR}/sysroot/usr" ] || error "Sysroot not found at ${OUTPUT_DIR}/sysroot. Run applications/onnxrt/ort/build.sh first."
    [ -f "${OUTPUT_DIR}/lib/libonnxruntime.so" ] || error "libonnxruntime.so not found at ${OUTPUT_DIR}/lib. Run applications/onnxrt/ort/build.sh first."
    [ -f "${RUNNER_DIR}/yolo_runner.cpp" ] || error "Runner source not found at ${RUNNER_DIR}"
    info "All prerequisites met."
}

check_prerequisites

# --- Common compilation flags ---
COMMON_FLAGS=(
    --target=riscv64-unknown-linux-gnu
    --sysroot="${OUTPUT_DIR}/sysroot"
    -march=rv64gcv
    -isystem "${OUTPUT_DIR}/sysroot/usr/include/riscv64-linux-gnu"
    -std=c++17
    -O2
    -g
    -fuse-ld=lld
    -I"${OUTPUT_DIR}/include/onnxruntime"
    -I"${OUTPUT_DIR}/include/onnxruntime/core/session"
)

SYSROOT_LIB="${OUTPUT_DIR}/sysroot/usr/lib"

# --- Build yolo_inference ---
build_yolo_inference() {
    local runner_out="${OUTPUT_DIR}/yolo_inference"

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "yolo_inference already built at ${runner_out}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling yolo_inference..."

    "${LLVM_INSTALL}/bin/clang++" \
        "${COMMON_FLAGS[@]}" \
        -I"${RUNNER_DIR}" \
        "${RUNNER_DIR}/yolo_runner.cpp" \
        -o "${runner_out}" \
        -L"${SYSROOT_LIB}" \
        -lonnxruntime

    info "yolo_inference built: ${runner_out}"
    file "${runner_out}"
}

build_yolo_inference

# --- Build yolo_preprocess ---
build_yolo_preprocess() {
    local runner_out="${OUTPUT_DIR}/yolo_preprocess"
    local src_file="${RUNNER_DIR}/yolo_preprocess.cpp"

    if [[ ! -f "${src_file}" ]]; then
        warn "yolo_preprocess.cpp not found, skipping"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "yolo_preprocess already built at ${runner_out}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling yolo_preprocess..."

    "${LLVM_INSTALL}/bin/clang++" \
        "${COMMON_FLAGS[@]}" \
        -I"${RUNNER_DIR}" \
        "${src_file}" \
        -o "${runner_out}" \
        -L"${SYSROOT_LIB}" \
        -lonnxruntime

    info "yolo_preprocess built: ${runner_out}"
    file "${runner_out}"
}

build_yolo_preprocess

# --- Build yolo_postprocess ---
build_yolo_postprocess() {
    local runner_out="${OUTPUT_DIR}/yolo_postprocess"
    local src_file="${RUNNER_DIR}/yolo_postprocess.cpp"

    if [[ ! -f "${src_file}" ]]; then
        warn "yolo_postprocess.cpp not found, skipping"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "yolo_postprocess already built at ${runner_out}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling yolo_postprocess..."

    "${LLVM_INSTALL}/bin/clang++" \
        "${COMMON_FLAGS[@]}" \
        -I"${RUNNER_DIR}" \
        "${src_file}" \
        -o "${runner_out}" \
        -L"${SYSROOT_LIB}" \
        -lonnxruntime

    info "yolo_postprocess built: ${runner_out}"
    file "${runner_out}"
}

build_yolo_postprocess

# --- Build generic_ort_runner ---
build_generic_ort_runner() {
    local runner_out="${OUTPUT_DIR}/generic_ort_runner"
    local src_file="${RUNNER_DIR}/generic_ort_runner.cpp"

    if [[ ! -f "${src_file}" ]]; then
        warn "generic_ort_runner.cpp not found, skipping"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -f "${runner_out}" ]]; then
        info "generic_ort_runner already built at ${runner_out}. Use --force to rebuild."
        return 0
    fi

    info "Cross-compiling generic_ort_runner..."

    "${LLVM_INSTALL}/bin/clang++" \
        "${COMMON_FLAGS[@]}" \
        -fno-omit-frame-pointer \
        "${src_file}" \
        -o "${runner_out}" \
        -L"${SYSROOT_LIB}" \
        -lonnxruntime

    info "generic_ort_runner built: ${runner_out}"
    file "${runner_out}"
}

build_generic_ort_runner

# --- Build minimal rootfs for chroot profiling ---
build_rootfs() {
    local sysroot="${OUTPUT_DIR}/sysroot"
    local rootfs="${OUTPUT_DIR}/rootfs"
    local tarball="${OUTPUT_DIR}/rootfs.tar.gz"
    local sysroot_lib="${sysroot}/usr/lib/riscv64-linux-gnu"

    if [[ "${FORCE}" != "true" && -f "${tarball}" ]]; then
        info "rootfs.tar.gz already exists at ${tarball}. Use --force to rebuild."
        return 0
    fi

    [ -d "${sysroot_lib}" ] || { warn "Sysroot not found, skipping rootfs build"; return 0; }
    [ -f "${OUTPUT_DIR}/generic_ort_runner" ] || { warn "generic_ort_runner not found, skipping rootfs build"; return 0; }

    info "Building minimal rootfs for chroot profiling..."
    rm -rf "${rootfs}"
    mkdir -p "${rootfs}"/{lib,proc,dev,sys,tmp}

    # Runtime libraries — use -L to follow symlinks (sysroot has relative symlinks)
    local libs=(
        ld-linux-riscv64-lp64d.so.1
        libc.so.6 libc-*.so
        libm.so.6 libm-*.so
        libdl.so.2 libdl-*.so
        librt.so.1 librt-*.so
        libpthread.so.0 libpthread-*.so
        libgcc_s.so.1
    )
    for lib in "${libs[@]}"; do
        for f in "${sysroot_lib}/${lib}"; do
            [ -e "$f" ] && cp -L "$f" "${rootfs}/lib/"
        done
    done
    # ld-linux is in sysroot/lib/riscv64-linux-gnu/ (real file, not symlink)
    cp -L "${sysroot}/lib/riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" "${rootfs}/lib/"

    # libstdc++ (may have multiple versions)
    cp -L "${sysroot_lib}"/libstdc++.so* "${rootfs}/lib/" 2>/dev/null || true

    # ORT libraries
    cp -a "${OUTPUT_DIR}/lib"/libonnxruntime.so* "${rootfs}/lib/" 2>/dev/null || true

    # Runner binary
    cp "${OUTPUT_DIR}/generic_ort_runner" "${rootfs}/"

    # Pack tarball
    tar czf "${tarball}" -C "${OUTPUT_DIR}" rootfs/
    local size
    size=$(du -sh "${tarball}" | cut -f1)

    info "rootfs built: ${tarball} (${size})"
    echo "  Libraries: $(ls "${rootfs}/lib/" | wc -l) files"
}

build_rootfs

# --- Done ---
info "All done!"
echo ""
echo "Artifacts:"
echo "  yolo_inference:   ${OUTPUT_DIR}/yolo_inference"
echo "  yolo_preprocess:  ${OUTPUT_DIR}/yolo_preprocess"
echo "  yolo_postprocess: ${OUTPUT_DIR}/yolo_postprocess"
echo "  generic_ort_runner: ${OUTPUT_DIR}/generic_ort_runner"
echo "  rootfs.tar.gz:    ${OUTPUT_DIR}/rootfs.tar.gz"
echo ""
file "${OUTPUT_DIR}/yolo_inference" || true
file "${OUTPUT_DIR}/yolo_preprocess" 2>/dev/null || true
file "${OUTPUT_DIR}/yolo_postprocess" 2>/dev/null || true
file "${OUTPUT_DIR}/generic_ort_runner" || true
ls -lh "${OUTPUT_DIR}/rootfs.tar.gz" 2>/dev/null || true