#!/bin/bash

# ============================================================================
# LAMMPS 编译脚本 - 集成 NEP 和 tabGAP 插件
# ============================================================================
# 
# 功能：编译带有 NEP 和 tabGAP 接口的 LAMMPS
# 输出：可执行文件将放置在 /Users/amphilagusgu/workspace/comparison_nep-gap/opt
#
# 使用方法：
#   chmod +x build_lammps_with_nep_tabgap.sh
#   ./build_lammps_with_nep_tabgap.sh
#
# ============================================================================

set -e  # 遇到错误立即退出

# 定义路径
WORKSPACE_ROOT="/Users/amphilagusgu/workspace/comparison_nep-gap"
LAMMPS_ROOT="${WORKSPACE_ROOT}/src/lammps"
LAMMPS_SRC="${LAMMPS_ROOT}/src"
NEP_INTERFACE="${WORKSPACE_ROOT}/src/nep_interface"
TABGAP_INTERFACE="${WORKSPACE_ROOT}/src/tabgap_interface"
BUILD_DIR="${LAMMPS_ROOT}/build"
OPT_DIR="${WORKSPACE_ROOT}/opt"

echo "============================================"
echo "开始编译 LAMMPS (NEP + tabGAP)"
echo "============================================"

# 创建输出目录
mkdir -p "${OPT_DIR}"

# ============================================================================
# 步骤 0: 检查依赖库
# ============================================================================
echo ""
echo "[0/6] 检查系统依赖库..."

# 检查是否安装了必要的数学库
check_library() {
    local lib_name=$1
    if brew list "$lib_name" &>/dev/null; then
        echo "  ✓ $lib_name 已安装"
        return 0
    else
        echo "  ✗ $lib_name 未安装"
        return 1
    fi
}

missing_libs=()

# 检查关键数学库
for lib in fftw openblas lapack open-mpi; do
    if ! check_library "$lib"; then
        missing_libs+=("$lib")
    fi
done

# 如果有缺失的库，提示用户安装
if [ ${#missing_libs[@]} -gt 0 ]; then
    echo ""
    echo "  警告：检测到缺失的依赖库，建议安装："
    echo "  brew install ${missing_libs[*]}"
    echo ""
    read -p "  是否继续编译？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  编译已取消"
        exit 1
    fi
fi

echo "  ✓ 依赖检查完成"

# ============================================================================
# 步骤 1: 集成 NEP 接口
# ============================================================================
echo ""
echo "[1/6] 集成 NEP 接口..."

# 创建 USER-NEP 目录
NEP_DIR="${LAMMPS_SRC}/USER-NEP"
mkdir -p "${NEP_DIR}"

# 复制 NEP 核心文件
echo "  - 复制 NEP 核心文件到 ${NEP_DIR}"
cp "${NEP_INTERFACE}/src/nep.h" "${NEP_DIR}/"
cp "${NEP_INTERFACE}/src/nep.cpp" "${NEP_DIR}/"
cp "${NEP_INTERFACE}/src/dftd3para.h" "${NEP_DIR}/"

# 复制 NEP-LAMMPS 接口文件
echo "  - 复制 NEP-LAMMPS 接口文件"
cp "${NEP_INTERFACE}/interface/lammps/USER-NEP/pair_NEP.h" "${NEP_DIR}/"
cp "${NEP_INTERFACE}/interface/lammps/USER-NEP/pair_NEP.cpp" "${NEP_DIR}/"

if [ -f "${NEP_INTERFACE}/interface/lammps/USER-NEP/Install.sh" ]; then
    cp "${NEP_INTERFACE}/interface/lammps/USER-NEP/Install.sh" "${NEP_DIR}/"
fi

# 创建 CMakeLists.txt 以便 CMake 能识别 USER-NEP 包
echo "  - 创建 USER-NEP 的 CMakeLists.txt"
cat > "${NEP_DIR}/CMakeLists.txt" << 'EOF'
# USER-NEP package for LAMMPS

# Add pair_nep source files
target_sources(lammps PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pair_NEP.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/nep.cpp
)

# Add include directory
target_include_directories(lammps PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
EOF

# 复制 USER-NEP.cmake 到 LAMMPS cmake/Modules/Packages 目录
echo "  - 复制 USER-NEP.cmake 到 cmake/Modules/Packages"
cp "${NEP_INTERFACE}/interface/lammps/USER-NEP.cmake" "${LAMMPS_ROOT}/cmake/Modules/Packages/"

# 修改 LAMMPS cmake/CMakeLists.txt 注册 USER-NEP 包
echo "  - 修改 LAMMPS cmake/CMakeLists.txt 注册 USER-NEP 包"
CMAKE_FILE="${LAMMPS_ROOT}/cmake/CMakeLists.txt"

# 备份原始文件（只备份一次）
if [ ! -f "${CMAKE_FILE}.backup" ]; then
    cp "${CMAKE_FILE}" "${CMAKE_FILE}.backup"
    echo "  - 已备份原始 CMakeLists.txt"
fi

# 检查是否已经添加过 USER-NEP
if ! grep -q "USER-NEP" "${CMAKE_FILE}"; then
    # 在 STANDARD_PACKAGES 列表中添加 USER-NEP（按字母顺序插入）
    # 在 YAFF 之前插入
    sed -i.tmp '/^  YAFF)/i\
  USER-NEP' "${CMAKE_FILE}"
    
    rm -f "${CMAKE_FILE}.tmp"
    echo "  ✓ USER-NEP 已注册到 LAMMPS CMake 系统"
else
    echo "  ✓ USER-NEP 已存在于 CMakeLists.txt 中"
fi

echo "  ✓ NEP 接口集成完成"

# ============================================================================
# 步骤 2: 集成 tabGAP 接口
# ============================================================================
echo ""
echo "[2/6] 集成 tabGAP 接口..."

# 复制 tabGAP 文件到 LAMMPS src 目录
echo "  - 复制 pair_tabgap.* 文件到 ${LAMMPS_SRC}"
cp "${TABGAP_INTERFACE}/lammps/pair_tabgap.cpp" "${LAMMPS_SRC}/"
cp "${TABGAP_INTERFACE}/lammps/pair_tabgap.h" "${LAMMPS_SRC}/"

# 如果有 KOKKOS 版本，也复制
if [ -d "${TABGAP_INTERFACE}/lammps/KOKKOS" ]; then
    echo "  - 复制 KOKKOS 版本文件"
    cp -r "${TABGAP_INTERFACE}/lammps/KOKKOS/"* "${LAMMPS_SRC}/KOKKOS/" 2>/dev/null || true
fi

echo "  ✓ tabGAP 接口集成完成"

# ============================================================================
# 步骤 2.5: 清理残留文件
# ============================================================================
echo ""
echo "[2.5/7] 清理残留文件..."

# 清理 LAMMPS 构建目录（标准做法：删除整个 build 目录）
if [ -d "${BUILD_DIR}" ]; then
    echo "  - 删除旧的构建目录: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

# 重新创建干净的构建目录
echo "  - 创建新的构建目录: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo "  ✓ 清理完成"

# ============================================================================
# 步骤 3: 配置 CMake
# ============================================================================
echo ""
echo "[3/7] 配置 CMake..."

# 进入构建目录
cd "${BUILD_DIR}"

# 检测 Homebrew 安装的库路径（macOS）
# 注意：OpenBLAS 和 LAPACK 是 keg-only 的，需要显式指定路径
if command -v brew &>/dev/null; then
    FFTW_ROOT=$(brew --prefix fftw 2>/dev/null || echo "")
    OPENBLAS_ROOT=$(brew --prefix openblas 2>/dev/null || echo "")
    LAPACK_ROOT=$(brew --prefix lapack 2>/dev/null || echo "")
fi

# 设置编译器环境变量以找到 keg-only 库
# OpenBLAS 和 LAPACK 在 macOS 上是 keg-only 的，不会自动链接
export LDFLAGS=""
export CPPFLAGS=""

if [ -n "$OPENBLAS_ROOT" ]; then
    export LDFLAGS="${LDFLAGS} -L${OPENBLAS_ROOT}/lib"
    export CPPFLAGS="${CPPFLAGS} -I${OPENBLAS_ROOT}/include"
    echo "  - 配置 OpenBLAS 环境变量: ${OPENBLAS_ROOT}"
fi

if [ -n "$LAPACK_ROOT" ]; then
    export LDFLAGS="${LDFLAGS} -L${LAPACK_ROOT}/lib"
    export CPPFLAGS="${CPPFLAGS} -I${LAPACK_ROOT}/include"
    echo "  - 配置 LAPACK 环境变量: ${LAPACK_ROOT}"
fi

# 运行 CMake 配置
# 注意：
# - tabGAP 需要 MANYBODY 包来支持 EAM 部分
# - NEP 接口支持 MPI 并行
# - OpenMP 已禁用（macOS AppleClang 不支持）
# - 启用 FFTW3、BLAS、LAPACK 等数学库
# - macOS 可以选择使用 Accelerate.framework（系统自带）或 Homebrew 的库
echo "  - 运行 CMake 配置（启用 MPI 和数学库，OpenMP 已禁用）"

CMAKE_ARGS=(
    "${LAMMPS_ROOT}/cmake"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_FLAGS="-ffast-math -march=native"
    -DPKG_MANYBODY=yes
    -DPKG_USER-NEP=yes
    -DBUILD_MPI=yes
    -DBUILD_OMP=no
    -DFFT=FFTW3
    -DPKG_KSPACE=yes
)

echo "  - 启用包: MANYBODY, USER-NEP, KSPACE"

# 如果找到了 FFTW，添加路径
if [ -n "$FFTW_ROOT" ]; then
    CMAKE_ARGS+=(-DFFTW3_INCLUDE_DIR="${FFTW_ROOT}/include")
    CMAKE_ARGS+=(-DFFTW3_LIBRARY="${FFTW_ROOT}/lib/libfftw3.dylib")
    echo "  - 使用 FFTW3: ${FFTW_ROOT}"
fi

# 选择 BLAS/LAPACK 实现
# 选项 1: 使用 macOS 自带的 Accelerate.framework（推荐，性能好且无需额外配置）
# 选项 2: 使用 Homebrew 的 OpenBLAS/LAPACK（如果需要特定版本）
USE_ACCELERATE=true  # 改为 false 可使用 Homebrew 版本

if [ "$USE_ACCELERATE" = true ]; then
    # 使用 macOS Accelerate.framework（Apple 优化的 BLAS/LAPACK）
    CMAKE_ARGS+=(-DBLA_VENDOR=Apple)
    echo "  - 使用 macOS Accelerate.framework (Apple 优化的 BLAS/LAPACK)"
else
    # 使用 Homebrew 安装的 OpenBLAS/LAPACK
    if [ -n "$OPENBLAS_ROOT" ]; then
        CMAKE_ARGS+=(-DBLAS_LIBRARIES="${OPENBLAS_ROOT}/lib/libopenblas.dylib")
        echo "  - 使用 OpenBLAS: ${OPENBLAS_ROOT}"
    fi
    
    if [ -n "$LAPACK_ROOT" ]; then
        CMAKE_ARGS+=(-DLAPACK_LIBRARIES="${LAPACK_ROOT}/lib/liblapack.dylib")
        echo "  - 使用 LAPACK: ${LAPACK_ROOT}"
    fi
fi

CMAKE_ARGS+=(-DCMAKE_INSTALL_PREFIX="${OPT_DIR}")

cmake "${CMAKE_ARGS[@]}"

echo "  ✓ CMake 配置完成"

# ============================================================================
# 步骤 4: 编译 LAMMPS
# ============================================================================
echo ""
echo "[4/7] 编译 LAMMPS..."

# 使用多核编译（根据系统调整 -j 参数）
NCORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "  - 使用 ${NCORES} 个核心进行编译"

cmake --build . -- -j${NCORES}

echo "  ✓ 编译完成"

# ============================================================================
# 步骤 5: 安装到 opt 目录
# ============================================================================
echo ""
echo "[5/7] 安装可执行文件到 ${OPT_DIR}..."

# 复制可执行文件
if [ -f "lmp" ]; then
    cp lmp "${OPT_DIR}/lmp_nep_tabgap"
    echo "  ✓ 可执行文件已复制到: ${OPT_DIR}/lmp_nep_tabgap"
else
    echo "  ✗ 错误：找不到编译后的可执行文件 'lmp'"
    exit 1
fi

# ============================================================================
# 步骤 6: 验证编译结果
# ============================================================================
echo ""
echo "[6/7] 验证编译结果..."

# 检查可执行文件是否可以运行
if "${OPT_DIR}/lmp_nep_tabgap" -help &>/dev/null; then
    echo "  ✓ 可执行文件验证成功"
else
    echo "  ⚠ 警告：可执行文件可能无法正常运行，请检查依赖库"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "============================================"
echo "编译完成！"
echo "============================================"
echo ""
echo "可执行文件位置: ${OPT_DIR}/lmp_nep_tabgap"
echo ""
echo "运行方法："
echo "  串行运行："
echo "    ${OPT_DIR}/lmp_nep_tabgap -in input.lammps"
echo ""
echo "  MPI 并行运行（例如 4 个进程）："
echo "    mpirun -np 4 ${OPT_DIR}/lmp_nep_tabgap -in input.lammps"
echo ""
echo "势能配置示例："
echo "  NEP 势能："
echo "    pair_style nep"
echo "    pair_coeff * * /path/to/nep.txt C"
echo ""
echo "  tabGAP 势能（2b+3b）："
echo "    pair_style tabgap"
echo "    pair_coeff * * /path/to/potential.tabgap W"
echo ""
echo "  tabGAP 势能（2b+3b+EAM）："
echo "    pair_style hybrid/overlay tabgap eam/fs"
echo "    pair_coeff * * tabgap /path/to/potential.tabgap W"
echo "    pair_coeff * * eam/fs /path/to/potential.eam.fs W"
echo ""
echo "============================================"
