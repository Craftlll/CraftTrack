#!/bin/bash

# 1. 检查环境
current_env=$(conda info --envs | grep "*" | awk '{print $1}')
if [ "$current_env" != "mtrack" ]; then
    echo "错误: 当前激活的环境是 '$current_env'。"
    echo "请先激活 mtrack 环境: conda activate mtrack"
    exit 1
fi

echo ">>> 正在配置 mtrack 环境以适配 ARTrack-Mamba..."

# 2. 安装/检查 Mamba (这是你的创新点核心)
echo ">>> [1/4] 检查 Mamba 环境..."
if python -c "import mamba_ssm; import causal_conv1d" &> /dev/null; then
    echo "    Mamba (mamba_ssm & causal_conv1d) 已安装，跳过。"
else
    echo "    警告: 检测到未安装 mamba_ssm。"
    echo "    请确保你已安装适合当前 CUDA 版本的 mamba_ssm。"
    echo "    建议命令: pip install causal-conv1d>=1.2.0 mamba-ssm"
    # 这里不自动安装，防止版本错乱，留给用户手动确认
fi

# 3. 安装 ARTrack 工具包 (不升级现有包，避免破坏环境)
echo ">>> [2/4] 安装 ARTrack 依赖工具库..."
pip install -r requirements_artrack_adapter.txt --no-deps

# 4. 特殊处理 TIMM
# ARTrack 代码使用了 timm 0.5.4 的旧版导入方式 (from timm.models.layers import ...)
# 新版 timm (0.9+) 改变了结构。为了保证代码能跑，我们需要确认 timm 版本。
echo ">>> [3/4] 检查 timm 版本兼容性..."
pip install "timm==0.5.4" 
# 注意：如果这步报错（例如因为 python 版本过高），可以尝试 pip install timm==0.6.13
# 只要不安装 0.9.x 以上版本，通常兼容性较好。

# 5. 编译 PreciseRoIPooling (ARTrack 必须组件)
# 必须使用当前的 PyTorch 和 CUDA 重新编译，否则会报符号错误
echo ">>> [4/4] 编译 PreciseRoIPooling (CUDA 算子)..."
if [ -d "lib/models/artrackv2_mindspore/external/PreciseRoIPooling" ]; then
    cd lib/models/artrackv2_mindspore/external/PreciseRoIPooling
    
    # 清理旧的构建文件
    rm -rf build/ dist/ *.egg-info
    
    # 编译安装
    python setup.py install
    
    cd ../../../../..
    echo "    PreciseRoIPooling 编译完成。"
else
    echo "    警告: 未找到 PreciseRoIPooling 目录，跳过编译。"
    echo "    请检查代码路径: lib/models/artrackv2_mindspore/external/PreciseRoIPooling"
fi

echo ">>> 配置完成！"
echo "请运行: python test_artrack_mamba.py 再次验证。"