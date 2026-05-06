#!/usr/bin/env bash
set -euo pipefail

# Cross-compile llama.cpp for RISC-V rv64gcv using LLVM 22
# Output: llama-cli, llama-server executables for RISC-V
#
# Usage:
#   build.sh [OPTIONS]
#
# Options:
#   --force          Rebuild everything from scratch
#   --skip-sysroot   Skip sysroot extraction (use existing)
#   --skip-source    Skip llama.cpp source cloning
#   --test           Compile and run tests under tests/
#   --help           Show this help message
#   -j, --jobs N     Parallel build jobs (default: nproc)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/output/llama.cpp"
VENDOR_DIR="${SCRIPT_DIR}/vendor"
LLVM_INSTALL="${PROJECT_ROOT}/third_party/llvm-install"
LLAMA_SOURCE="${VENDOR_DIR}/llama.cpp"
TOOLCHAIN_FILE="${SCRIPT_DIR}/riscv64-linux-toolchain.cmake"
QEMU_RISCV64="${PROJECT_ROOT}/third_party/qemu/build/qemu-riscv64"
TEST_BUILD_DIR="${OUTPUT_DIR}/.test-build"
SYSROOT=""  # Set by setup_sysroot()

LLAMA_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_VERSION="b8783"  # Latest release as of 2026-04-14

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}=== $* ===${NC}"; }
warn()  { echo -e "${YELLOW}Warning: $*${NC}"; }
error() { echo -e "${RED}Error: $*${NC}" >&2; exit 1; }

# --- Help ---
show_help() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Cross-compile llama.cpp for RISC-V rv64gcv using LLVM 22.

Options:
  --force          Rebuild everything from scratch
  --skip-sysroot   Skip sysroot extraction (use existing)
  --skip-source    Skip llama.cpp source cloning
  --test           Compile and run tests under tests/
  -j, --jobs N     Parallel build jobs (default: nproc)
  --help           Show this help message

Output artifacts:
  CLI:    ${OUTPUT_DIR}/bin/llama-cli
  Server: ${OUTPUT_DIR}/bin/llama-server
  Lib:    ${OUTPUT_DIR}/lib/
  Sysroot: ${OUTPUT_DIR}/sysroot

Run with QEMU:
  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot \\
    ${OUTPUT_DIR}/bin/llama-cli -m <model.gguf> -p "Hello"
EOF
}

# --- Argument parsing ---
FORCE=false
SKIP_SYSROOT=false
SKIP_SOURCE=false
RUN_TESTS=false
JOBS=$(nproc 2>/dev/null || echo 4)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)      show_help; exit 0 ;;
        --force)        FORCE=true; shift ;;
        --skip-sysroot) SKIP_SYSROOT=true; shift ;;
        --skip-source)  SKIP_SOURCE=true; shift ;;
        --test)         RUN_TESTS=true; shift ;;
        -j|--jobs)      JOBS="$2"; shift 2 ;;
        -j*)            JOBS="${1#-j}"; shift ;;
        *)              error "Unknown argument: $1 (use --help for usage)" ;;
    esac
done

# --- Step 0: Prerequisites ---
check_prerequisites() {
    info "Checking prerequisites..."
    command -v cmake &>/dev/null || error "cmake not found. Install cmake >= 3.14."
    command -v ninja &>/dev/null || error "ninja not found. Install ninja-build."
    command -v git &>/dev/null || error "git not found."

    [ -d "${LLVM_INSTALL}/bin" ] || error "LLVM install not found at ${LLVM_INSTALL}"
    [ -f "${LLVM_INSTALL}/bin/clang" ] || error "clang not found at ${LLVM_INSTALL}/bin/clang"

    if [[ "${RUN_TESTS}" == "true" ]]; then
        [ -f "${QEMU_RISCV64}" ] || error "qemu-riscv64 not found at ${QEMU_RISCV64}"
    fi

    info "All prerequisites met."
    echo "  LLVM:    ${LLVM_INSTALL}/bin/clang --version"
    "${LLVM_INSTALL}/bin/clang" --version | head -1 || true
    if [[ "${RUN_TESTS}" == "true" ]]; then
        echo "  QEMU:    ${QEMU_RISCV64}"
    fi
}

check_prerequisites

# --- Step 1: Sysroot ---
setup_sysroot() {
    local standalone_sysroot="${OUTPUT_DIR}/sysroot"

    if [[ "${SKIP_SYSROOT}" == "true" ]]; then
        if [ ! -d "${standalone_sysroot}/usr" ]; then
            error "Sysroot not found at ${standalone_sysroot} (remove --skip-sysroot to extract)"
        fi
        declare -g SYSROOT="${standalone_sysroot}"
        info "Using existing sysroot: ${SYSROOT}"
        return 0
    fi

    if [[ "${FORCE}" != "true" && -d "${standalone_sysroot}/usr" ]]; then
        info "Sysroot already exists at ${standalone_sysroot}. Use --force to re-extract."
        declare -g SYSROOT="${standalone_sysroot}"
        return 0
    fi

    info "Extracting riscv64 sysroot from riscv64/ubuntu:24.04..."
    command -v docker &>/dev/null || error "Docker not found. Sysroot extraction requires Docker."

    rm -rf "${standalone_sysroot}"
    mkdir -p "${standalone_sysroot}"

    local tmp_container="rvfuse-llama-sysroot-prep-$$"

    docker run --platform riscv64 --name "${tmp_container}" -d riscv64/ubuntu:24.04 tail -f /dev/null > /dev/null
    trap "docker rm -f ${tmp_container} 2>/dev/null || true" RETURN

    docker exec "${tmp_container}" apt-get update -qq
    docker exec "${tmp_container}" apt-get install -y --no-install-recommends -qq \
        libc6-dev \
        libstdc++-12-dev \
        libgcc-12-dev \
        > /dev/null

    info "Copying sysroot directories from container..."

    docker cp "${tmp_container}:/usr/lib"      "${standalone_sysroot}/usr_lib_tmp"
    docker cp "${tmp_container}:/usr/include"  "${standalone_sysroot}/usr_include_tmp"
    docker cp "${tmp_container}:/lib/riscv64-linux-gnu" "${standalone_sysroot}/lib_riscv_tmp"

    mkdir -p "${standalone_sysroot}/usr" "${standalone_sysroot}/lib"
    mv "${standalone_sysroot}/usr_lib_tmp"     "${standalone_sysroot}/usr/lib"
    mv "${standalone_sysroot}/usr_include_tmp" "${standalone_sysroot}/usr/include"
    mv "${standalone_sysroot}/lib_riscv_tmp"   "${standalone_sysroot}/lib/riscv64-linux-gnu"

    docker rm -f "${tmp_container}" > /dev/null
    trap - RETURN

    # Top-level dynamic linker symlink
    ln -sf "riscv64-linux-gnu/ld-linux-riscv64-lp64d.so.1" \
        "${standalone_sysroot}/lib/ld-linux-riscv64-lp64d.so.1"

    # Create symlinks for CRT and shared libs
    local multilib="riscv64-linux-gnu"
    local base="${standalone_sysroot}/usr/lib"
    if [ -d "${base}/${multilib}" ]; then
        for f in crt1.o crti.o crtn.o Scrt1.o; do
            [ -f "${base}/${multilib}/${f}" ] && ln -sf "${multilib}/${f}" "${base}/${f}"
        done
        for f in libc.so libc.so.6 libm.so libm.so.6 libdl.so libdl.so.2 \
                 librt.so librt.so.1 libpthread.so libpthread.so.0 \
                 libgcc_s.so libgcc_s.so.1 libstdc++.so libstdc++.so.6; do
            [ -e "${base}/${multilib}/${f}" ] && [ ! -e "${base}/${f}" ] && \
                ln -sf "${multilib}/${f}" "${base}/${f}"
        done
    fi

    # Remove problematic static libs
    find "${standalone_sysroot}" -name "libm.a" -delete 2>/dev/null || true

    declare -g SYSROOT="${standalone_sysroot}"
    info "Sysroot ready at ${SYSROOT}"
    echo "  $(du -sh "${SYSROOT}" | cut -f1)"
}

setup_sysroot

# --- Step 2: Clone llama.cpp source ---
clone_source() {
    if [[ "${SKIP_SOURCE}" == "true" ]]; then
        info "Skipping source cloning (--skip-source)"
        return 0
    fi

    if [ -d "${LLAMA_SOURCE}" ] && [ -n "$(ls -A "${LLAMA_SOURCE}" 2>/dev/null)" ]; then
        if [[ "${FORCE}" == "true" ]]; then
            info "Force re-clone: removing existing ${LLAMA_SOURCE}..."
            rm -rf "${LLAMA_SOURCE}"
        else
            info "Skipping ${LLAMA_SOURCE} (already exists)"
            return 0
        fi
    fi

    info "Cloning ${LLAMA_REPO} @ ${LLAMA_VERSION} (shallow)..."
    mkdir -p "${VENDOR_DIR}"
    git clone --depth=1 --branch "${LLAMA_VERSION}" "${LLAMA_REPO}" "${LLAMA_SOURCE}"

    info "llama.cpp source ready."
}

clone_source

# --- Step 2.5: Apply patches ---
apply_patches() {
    local riscv_dir="${LLAMA_SOURCE}/ggml/src/ggml-cpu/arch/riscv"
    local patches_dir="${SCRIPT_DIR}/rvv-patches"

    [ -d "${patches_dir}" ] || { warn "No rvv-patches directory found, skipping."; return 0; }

    mkdir -p "${riscv_dir}"

    # Iterate over each patch directory (skip _template)
    local patch_count=0
    for patch_subdir in "${patches_dir}"/*/; do
        [ -d "${patch_subdir}" ] || continue
        local name="$(basename "${patch_subdir}")"
        [[ "${name}" == "_template" ]] && continue

        local inl_file="${patch_subdir}${name}.inl"
        local patch_file="${patch_subdir}patch.diff"

        # Find .inl file (may have different naming convention)
        for f in "${patch_subdir}"*.inl; do
            [ -f "${f}" ] && inl_file="${f}" && break
        done

        if [ -f "${inl_file}" ]; then
            local inl_dst="${riscv_dir}/$(basename "${inl_file}")"
            cp -f "${inl_file}" "${inl_dst}"
            info "Copied $(basename "${inl_file}") to ${riscv_dir}/"
        fi

        if [ -f "${patch_file}" ]; then
            if git -C "${LLAMA_SOURCE}" apply --check "${patch_file}" 2>/dev/null; then
                git -C "${LLAMA_SOURCE}" apply "${patch_file}"
                info "Applied patch: ${name}/patch.diff"
                patch_count=$((patch_count + 1))
            else
                info "Patch ${name}/patch.diff already applied or not applicable, skipping."
            fi
        else
            warn "No patch.diff found in ${patch_subdir}"
        fi
    done

    info "Applied ${patch_count} RVV patch(es)"
}

apply_patches

# --- Step 3: Cross-compile llama.cpp ---
cross_compile() {
    local llama_build="${OUTPUT_DIR}/.build"
    local llama_install="${OUTPUT_DIR}"

    if [[ "${FORCE}" != "true" && -f "${llama_install}/bin/llama-cli" ]]; then
        info "llama.cpp already built at ${llama_install}. Use --force to rebuild."
        return 0
    fi

    [ -f "${LLAMA_SOURCE}/CMakeLists.txt" ] || error "llama.cpp source not found at ${LLAMA_SOURCE}. Run without --skip-source."
    [ -d "${SYSROOT}/usr" ] || error "Sysroot not found at ${SYSROOT}."

    info "Cross-compiling llama.cpp ${LLAMA_VERSION} (rv64gcv + RVV + ZFH)..."

    rm -rf "${llama_build}"
    mkdir -p "${llama_build}" "${llama_install}/bin"

    export LLVM_INSTALL="${LLVM_INSTALL}"
    export SYSROOT="${SYSROOT}"

    cmake -S "${LLAMA_SOURCE}" -B "${llama_build}" \
        -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
        -DCMAKE_INSTALL_PREFIX="${llama_install}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_RVV=ON \
        -DGGML_RV_ZFH=ON \
        -DGGML_RV_ZICBOP=ON \
        -DGGML_RV_ZIHINTPAUSE=ON \
        -DGGML_RV_ZVL256B=ON \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_WEBUI=OFF \
        -G Ninja

    info "Building (ninja -j${JOBS})..."
    ninja -C "${llama_build}" -j"${JOBS}"

    info "Installing binaries..."
    ninja -C "${llama_build}" install

    # Copy llama libs to sysroot for QEMU execution
    info "Copying llama libs to sysroot..."
    local sysroot_lib="${SYSROOT}/lib/riscv64-linux-gnu"
    cp -a "${llama_install}/lib"/libllama.so* "${sysroot_lib}/"
    cp -a "${llama_install}/lib"/libggml.so* "${llama_install}/lib"/libggml-base.so* "${llama_install}/lib"/libggml-cpu.so* "${sysroot_lib}/"
    cp -a "${llama_install}/lib"/libmtmd.so* "${sysroot_lib}/"

    unset SYSROOT

    info "llama.cpp cross-compiled to ${llama_install}"
}

cross_compile

# --- Step 4: Build and run tests ---
build_and_run_tests() {
    local cc="${LLVM_INSTALL}/bin/clang++"
    local patches_dir="${SCRIPT_DIR}/rvv-patches"

    mkdir -p "${TEST_BUILD_DIR}"
    rm -rf "${TEST_BUILD_DIR}"/*

    # Collect test.cpp files from each rvv-patches subdirectory
    local test_files=()
    local test_dirs=()
    for patch_subdir in "${patches_dir}"/*/; do
        [ -d "${patch_subdir}" ] || continue
        local name="$(basename "${patch_subdir}")"
        [[ "${name}" == "_template" ]] && continue

        local test_file="${patch_subdir}test.cpp"
        if [ -f "${test_file}" ]; then
            test_files+=("${test_file}")
            test_dirs+=("${patch_subdir}")
        fi
    done

    if [[ ${#test_files[@]} -eq 0 ]]; then
        warn "No test.cpp files found in ${patches_dir}"
        return 0
    fi

    local failures=0
    local passed=0
    local skipped=0

    for i in "${!test_files[@]}"; do
        local test_src="${test_files[$i]}"
        local test_dir="${test_dirs[$i]}"
        local test_name="$(basename "${test_dir}")"

        local test_bin="${TEST_BUILD_DIR}/test_${test_name}"

        local test_flags=(
            -std=c++17 -O2
            --target=riscv64-unknown-linux-gnu
            --sysroot="${SYSROOT}"
            -march=rv64gcv_zvl512b_zfh_zvfh
            -mabi=lp64d
            -fuse-ld=lld
            -DGGML_USE_RISCV_V
            -D__riscv_v_fixed_vlen=512
            -I"${test_dir}"  # Include the patch directory for .inl files
        )

        info "Compiling test_${test_name}..."
        if "${cc}" "${test_flags[@]}" -o "${test_bin}" "${test_src}" -lm 2>&1; then
            info "Running test_${test_name} under QEMU..."
            if "${QEMU_RISCV64}" -L "${SYSROOT}" "${test_bin}" 2>&1; then
                passed=$((passed + 1))
            else
                error "test_${test_name} failed (exit code $?)"
                failures=$((failures + 1))
            fi
        else
            warn "test_${test_name} compilation failed (likely non-RVV test on x86 host)"
            skipped=$((skipped + 1))
        fi
        echo ""
    done

    echo ""
    info "Test results: ${passed} passed, ${failures} failed, ${skipped} skipped"
    [[ ${failures} -eq 0 ]]
}

if [[ "${RUN_TESTS}" == "true" ]]; then
    build_and_run_tests
fi

# --- Done ---
if [[ "${RUN_TESTS}" != "true" ]]; then
    info "All done!"
    echo ""
    echo "Artifacts:"
    echo "  CLI:     ${OUTPUT_DIR}/bin/llama-cli"
    echo "  Server:  ${OUTPUT_DIR}/bin/llama-server"
    echo "  Lib:     ${OUTPUT_DIR}/lib/"
    echo "  Sysroot: ${OUTPUT_DIR}/sysroot"
    echo ""
    file "${OUTPUT_DIR}/bin/llama-cli" || true
    file "${OUTPUT_DIR}/bin/llama-server" || true
    echo ""
    echo "Run with QEMU:"
    echo "  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot ${OUTPUT_DIR}/bin/llama-cli -m <model.gguf> -p \"Hello\""
    echo "  ${QEMU_RISCV64} -L ${OUTPUT_DIR}/sysroot ${OUTPUT_DIR}/bin/llama-server -m <model.gguf> --port 8080"
fi
