#!/bin/bash

# ============================================================================
# 验证 NEP 接口是否正确安装到 LAMMPS
# ============================================================================

LAMMPS_BIN="/Users/amphilagusgu/workspace/comparison_nep-gap/opt/lmp_nep_tabgap"

echo "============================================"
echo "验证 LAMMPS NEP 接口安装"
echo "============================================"
echo ""

# 检查可执行文件是否存在
if [ ! -f "${LAMMPS_BIN}" ]; then
    echo "✗ 错误：找不到 LAMMPS 可执行文件: ${LAMMPS_BIN}"
    exit 1
fi

echo "✓ 找到 LAMMPS 可执行文件"
echo ""

# 检查是否包含 nep pair style
echo "检查可用的 pair styles..."
echo ""

# 创建临时输入文件
TMP_INPUT=$(mktemp)
cat > "${TMP_INPUT}" << 'EOF'
# 临时测试文件
units metal
atom_style atomic
region box block 0 1 0 1 0 1
create_box 1 box
pair_style nep
EOF

# 运行 LAMMPS 并捕获输出
OUTPUT=$(${LAMMPS_BIN} -in ${TMP_INPUT} 2>&1 || true)

# 清理临时文件
rm -f "${TMP_INPUT}"

# 检查输出
if echo "${OUTPUT}" | grep -q "Unknown pair style"; then
    echo "✗ NEP pair style 未找到"
    echo ""
    echo "错误信息："
    echo "${OUTPUT}" | grep -A 5 "Unknown pair style"
    echo ""
    echo "建议："
    echo "1. 检查 USER-NEP 是否在 CMakeLists.txt 中注册"
    echo "2. 确认编译时启用了 -DPKG_USER-NEP=yes"
    echo "3. 查看编译日志确认 USER-NEP 源文件被编译"
    exit 1
else
    echo "✓ NEP pair style 已成功安装！"
    echo ""
    echo "可以使用以下命令查看所有可用的 pair styles："
    echo "  ${LAMMPS_BIN} -h | grep -A 100 'Pair styles'"
fi

# 清理log文件
rm -f log.lammps

echo ""
echo "============================================"
echo "验证完成"
echo "============================================"
