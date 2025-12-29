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
echo ">>> 正在获取远程文件列表..."
remote_files_list=$(bypy list $REMOTE_DIR | grep "$EXT" | awk '{print $2}')

# 将文件列表转换为数组
remote_files=($remote_files_list)
total_files=${#remote_files[@]}

if [ $total_files -eq 0 ]; then
    echo "❌ 错误: 无法在网盘路径 $REMOTE_DIR 下找到文件，请检查路径是否正确。"
    exit 1
fi

echo ">>> 共找到 $total_files 个文件，开始处理..."
start_time=$(date +%s)
processed_count=0

for file in "${remote_files[@]}"; do
    ((processed_count++))
    
    # 计算进度和时间
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    # 计算百分比
    percent=$((processed_count * 100 / total_files))
    
    # 计算预估剩余时间 (ETA)
    # 只有当处理过至少一个文件后才能计算平均速度，为了避免除以零，初始设为"计算中..."
    if [ $processed_count -gt 1 ] && [ $elapsed -gt 0 ]; then
        avg_time_per_file=$((elapsed * 1000 / (processed_count - 1))) # 毫秒精度太麻烦，用秒。
        # 实际上我们刚开始第N个文件。已完成 N-1 个。
        # 如果是第一个文件，elapsed 包含的是 list 的时间吗？不，start_time 在 list 之后。
        # 所以 elapsed 是处理前 (processed_count-1) 个文件的时间。
        
        files_remaining=$((total_files - processed_count + 1))
        
        # 使用当前已经消耗的时间 / (当前序号-1) * 剩余数量
        # 注意：如果是第一个文件，elapsed可能很小，且 processed_count-1=0。
        
        avg_speed=$(awk "BEGIN {print $elapsed / ($processed_count - 1)}")
        eta_seconds=$(awk "BEGIN {print int($avg_speed * ($total_files - $processed_count + 1))}")
        
        # 格式化 ETA 为 HH:MM:SS
        eta_h=$((eta_seconds / 3600))
        eta_m=$(( (eta_seconds % 3600) / 60 ))
        eta_s=$((eta_seconds % 60))
        eta_str=$(printf "%02d:%02d:%02d" $eta_h $eta_m $eta_s)
    else
        eta_str="计算中..."
    fi

    # 进度条可视化
    bar_size=20
    done_chars=$((percent * bar_size / 100))
    todo_chars=$((bar_size - done_chars))
    # 生成进度条字符串
    bar=$(printf "%${done_chars}s" | tr ' ' '#')
    empty=$(printf "%${todo_chars}s" | tr ' ' '-')
    
    echo "----------------------------------------------------------------"
    echo "[$bar$empty] $percent% | 进度: $processed_count/$total_files | 耗时: ${elapsed}s | 预计剩余: $eta_str"
    
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
    # 使用 aria2 加速下载
    bypy downfile "$REMOTE_DIR/$file" "$LOCAL_DIR/$file" --downloader aria2
    
    if [ $? -eq 0 ]; then
        echo "✨ 下载完成: $file"
    else
        echo "⚠️  失败: $file 下载出现问题，尝试下一个。"
    fi
done

echo ">>> 所有任务处理完毕。"