#!/bin/bash

# ================= 配置区域 =================
# 1. 百度网盘中 LaSOT 压缩包所在的文件夹路径 (bypy list 看到的路径)
REMOTE_DIR="LaSOT" 

# 2. 你本地存放 LaSOT 压缩包的绝对路径
LOCAL_DIR="/root/autodl-tmp/data/lasot"

# 3. 压缩包的后缀 (通常是 .zip 或 .tar.gz)
EXT=".zip"
# ===========================================

echo ">>> 开始检查 LaSOT 数据集下载状态..."
mkdir -p $LOCAL_DIR

# 获取远程文件列表并提取文件名
# 我们只关注以 .zip 结尾的文件
remote_files=$(bypy list $REMOTE_DIR | grep "$EXT" | awk '{print $2}')

if [ -z "$remote_files" ]; then
    echo "❌ 错误: 无法在网盘路径 $REMOTE_DIR 下找到文件，请检查路径是否正确。"
    exit 1
fi

for file in $remote_files; do
    # 检查本地是否存在该文件
    if [ -f "$LOCAL_DIR/$file" ]; then
        # 获取本地文件大小（可选，用于更严谨的判断）
        local_size=$(stat -c%s "$LOCAL_DIR/$file")
        if [ $local_size -gt 1048576 ]; then # 简单判断文件是否大于1MB，防止空包
            echo "✅ 跳过: $file 已存在于本地 $LOCAL_DIR"
            continue
        fi
    fi

    echo "🚀 正在下载新文件: $file ..."
    # 使用 bypy downfile 下载单个文件到指定目录
    # 注意：bypy downfile 的参数是远程完整路径和本地保存路径
    bypy downfile "$REMOTE_DIR/$file" "$LOCAL_DIR/$file"
    
    if [ $? -eq 0 ]; then
        echo "✨ 下载完成: $file"
    else
        echo "⚠️  失败: $file 下载出现问题，尝试下一个。"
    fi
done

echo ">>> 所有任务处理完毕。"