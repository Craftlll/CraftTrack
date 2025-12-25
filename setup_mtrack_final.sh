#!/bin/bash

# ================= 路径配置区域 =================
# 1. 你的项目根目录 (根据你提供的路径反推)
PROJECT_ROOT="/root/CraftTrack"

# 2. PreciseRoIPooling 的源码位置 
# (通常在 external/PreciseRoIPooling/pytorch)
ROI_POOL_DIR="${PROJECT_ROOT}/external/PreciseRoIPooling/pytorch"
# ===============================================

echo ">>> [Mamba-ARTrack] 开始环境配置 (针对 CraftTrack)..."
echo ">>> 项目根目录: $PROJECT_ROOT"

# -----------------------------------------------------------
# 1. 检查 Conda 环境
# -----------------------------------------------------------
current_env=$(conda info --envs | grep "*" | awk '{print $1}')
if [ "$current_env" != "mtrack" ]; then
    echo "❌ 错误: 当前激活的环境是 '$current_env'。"
    echo "👉 请先执行: conda activate mtrack"
    exit 1
fi

# -----------------------------------------------------------
# 2. 验证并安装 timm (兼容性关键)
# -----------------------------------------------------------
echo ">>> [1/3] 验证 timm 版本..."
# 尝试获取当前版本
current_timm=$(pip show timm | grep Version | awk '{print $2}')

if [ "$current_timm" == "0.5.4" ]; then
    echo "✅ timm 版本正确: $current_timm"
else
    echo "⚠️  当前 timm 版本: $current_timm (目标: 0.5.4)"
    echo "    正在强制重装 timm==0.5.4 ..."
    pip install timm==0.5.4
fi

# -----------------------------------------------------------
# 3. 编译 PreciseRoIPooling (核心修复)
# -----------------------------------------------------------
echo ">>> [2/3] 编译 PreciseRoIPooling..."
echo "    目标路径: $ROI_POOL_DIR"

if [ -d "$ROI_POOL_DIR" ]; then
    cd "$ROI_POOL_DIR"
    
    # 清理旧的编译缓存，防止由于之前编译失败导致残留
    echo "    清理旧构建文件 (build/, dist/)..."
    rm -rf build/ dist/ *.egg-info
    
    # 检查 setup.py
    if [ -f "setup.py" ]; then
        echo "    正在编译 (使用当前 mtrack 环境的 CUDA)..."
        
        # 使用 pip install . 这种现代方式安装，比 setup.py install 更稳定
        pip install .
        
        if [ $? -eq 0 ]; then
            echo "✅ PreciseRoIPooling 编译并安装成功！"
        else
            echo "❌ 编译失败。"
            echo "    提示: 如果 Mamba 模型不使用 Region 操作，你可以忽略此错误继续运行。"
        fi
    else
        echo "❌ 错误: 在 $ROI_POOL_DIR 下未找到 setup.py"
    fi
else
    echo "❌ 错误: 找不到目录 $ROI_POOL_DIR"
    echo "    请检查你的文件结构，确认 external 文件夹是否存在于 /root/CraftTrack/ 下。"
fi

# -----------------------------------------------------------
# 4. 最终验证
# -----------------------------------------------------------
echo ">>> [3/3] 最终验证..."
cd "$PROJECT_ROOT"

# 简单尝试导入
python -c "import PrRoIPool; print('✅ PrRoIPool 模块加载成功')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  警告: 无法直接导入 PrRoIPool。"
    echo "    不过没关系，请直接尝试运行测试脚本: python test_artrack_mamba.py"
else
    echo "🎉 环境配置成功。"
fi