#!/bin/bash

# 定义源目录和目标目录
SOURCE_DIR="/root/autodl-pub/GOT10k"
DEST_DIR="/root/autodl-tmp/data/got10k"

# 创建目标目录
echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

# 1. 解压 val_data.zip (内部包含 val/ 目录)
if [ -f "$SOURCE_DIR/val_data.zip" ]; then
    echo "Unzipping val_data.zip..."
    unzip -q -n "$SOURCE_DIR/val_data.zip" -d "$DEST_DIR"
else
    echo "Warning: val_data.zip not found."
fi

# 2. 解压 test_data.zip (内部包含 test/ 目录)
if [ -f "$SOURCE_DIR/test_data.zip" ]; then
    echo "Unzipping test_data.zip..."
    unzip -q -n "$SOURCE_DIR/test_data.zip" -d "$DEST_DIR"
else
    echo "Warning: test_data.zip not found."
fi

# 3. 解压 train_data 下的所有 zip 文件
# 由于 train_data 下的 zip 解压后直接是 GOT-10k_Train_XXXXXX 文件夹，没有 train/ 父目录
# 所以我们需要手动创建一个 train/ 目录，并将它们解压进去
TRAIN_DEST="$DEST_DIR/train"
echo "Creating train directory: $TRAIN_DEST"
mkdir -p "$TRAIN_DEST"

if [ -d "$SOURCE_DIR/train_data" ]; then
    echo "Processing train_data..."
    
    # 遍历 train_data 目录下的所有 zip 文件
    count=0
    total=$(ls "$SOURCE_DIR/train_data"/*.zip 2>/dev/null | wc -l)
    
    for zipfile in "$SOURCE_DIR/train_data"/*.zip; do
        if [ -f "$zipfile" ]; then
            count=$((count+1))
            filename=$(basename "$zipfile")
            echo "[$count/$total] Unzipping $filename to $TRAIN_DEST..."
            unzip -q -n "$zipfile" -d "$TRAIN_DEST"
        fi
    done
    
    # 复制 list.txt (如果存在)
    if [ -f "$SOURCE_DIR/train_data/list.txt" ]; then
        echo "Copying list.txt..."
        cp "$SOURCE_DIR/train_data/list.txt" "$TRAIN_DEST/"
    fi
else
    echo "Warning: train_data directory not found."
fi

echo "All tasks completed! Data is ready at $DEST_DIR"
