# Cross-compilation toolchain for RISC-V 64-bit using LLVM 22
# Target: rv64gcv (IMAFD_C_V), Linux, lp64d ABI
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)
SET(CMAKE_SYSTEM_PROCESSOR riscv64)

SET(CMAKE_C_COMPILER $ENV{LLVM_INSTALL}/bin/clang)
SET(CMAKE_CXX_COMPILER $ENV{LLVM_INSTALL}/bin/clang++)

SET(CMAKE_C_COMPILER_TARGET riscv64-unknown-linux-gnu)
SET(CMAKE_CXX_COMPILER_TARGET riscv64-unknown-linux-gnu)

SET(CMAKE_SYSROOT $ENV{SYSROOT})
SET(CMAKE_FIND_ROOT_PATH $ENV{SYSROOT})

# Clang doesn't automatically add triplet-specific include paths like GCC does.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${CMAKE_SYSROOT}/usr/include/riscv64-linux-gnu")

# Enable RVV 1.0 auto-vectorization (LLVM 22)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")

# Use lld for linking (no GCC cross-compiler dependency).
# Previous c920-onnxrt toolchain used GCC's ld to work around R_RISCV_ALIGN
# issues in older lld versions. LLVM 22's lld has mature RISC-V support and
# handles this correctly.
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
SET(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")

# Debug symbols: preserved for profiling and debugging (DFG analysis, BBV).
# Symbols allow analyze_bbv.py to resolve addresses to function names.
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
