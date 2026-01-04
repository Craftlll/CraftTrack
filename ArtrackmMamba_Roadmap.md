# ARTrackMamba 改进与优化路线图 (Roadmap)

本文档旨在规划 `ARTrackMamba` 项目的后续改进方向，从当前的 Baseline 版本逐步迈向高性能、高效率的终极形态。

## 阶段一：现有 Baseline 的稳固与验证 (当前阶段)

**目标**: 确保当前的 Stateless 版本 (FP32, 并行推理) 能够稳定运行，精度对齐 ARTrackV2，无明显 Bug。

1.  **训练收敛验证**:
    *   完成 30 Epoch 训练。
    *   验证 Loss 曲线是否正常（无 NaN，平滑下降）。
    *   确认 Checkpoint 能够被测试脚本正确加载。
2.  **基准测试 (Benchmark)**:
    *   在 GOT-10k Test, TrackingNet, LaSOT 等数据集上运行测试。
    *   **关键指标对比**: 与 ARTrackV2-Base 对比 `AO` (Average Overlap) 和 `SR` (Success Rate)。
    *   *预期*: 精度应持平或略低（因单向/双向扫描差异），但不能崩盘。
3.  **可视化分析**:
    *   从测试集中抽取 Fail Cases (跟丢的视频)。
    *   分析原因：是遮挡没处理好？还是快速运动跟不上？

---

## 阶段二：推理加速与状态化 (Stateful Inference) —— *核心改进点*

**目标**: 利用 Mamba 的 RNN 属性，实现 $O(1)$ 复杂度的流式推理，大幅提升 FPS。

1.  **单向扫描 (Causal Layout) 改造**:
    *   *背景*: 要想用 RNN 状态传递，必须保证模型是因果的（Causal），即 $Token_t$ 只能看 $t$ 之前的信息。
    *   *行动*: 在 `vim.py` 中引入 `causal=True` 选项。
        *   Template 和 History 部分可以是双向的（因为它们是已知的）。
        *   **Search 区域必须改为单向扫描**，或者设计特殊的 Mask。
2.  **KV Cache / State Cache 实现**:
    *   修改 `VimBlock`，增加 `step(x, state)` 接口。
    *   在 `test/tracker/artrackmamba_seq.py` 中，不再每次拼接 `[Template, History, Search]`。
    *   **新逻辑**:
        *   第一帧 (Init): 输入 `Template`，算出并缓存 `State_0`。
        *   第二帧 (Track): 输入 `Search`，读取 `State_0`，更新为 `State_1`。
3.  **FPS 冲刺**:
    *   对比 Stateless vs Stateful 的 FPS。预期应有显著提升（尤其是在长序列 `NUMBER > 6` 时）。

---

## 阶段三：架构微调与精度提升

**目标**: 针对 Mamba 特性优化架构，超越 Baseline 精度。

1.  **增强位置编码**:
    *   引入 **RoPE (Rotary Positional Embeddings)** 替换现有的绝对位置编码。Mamba 对 RoPE 的支持更好，能增强长序列的外推能力。
2.  **特殊的扫描路径 (Scan Trajectory)**:
    *   目前的 1D 扫描会破坏 2D 图像的空间邻域信息。
    *   尝试 **Zig-Zag Scan** 或 **Hilbert Curve Scan**，让 1D 序列更好地保留 2D 空间结构。
3.  **混合架构 (Hybrid)**:
    *   尝试 `Mamba-Attention` 混合。在深层插入 1-2 层 Standard Attention，用来“校准”长期记忆，找回丢失的全局信息。

---

## 阶段四：工程化落地

1.  **TensorRT 部署**:
    *   Mamba 的算子 (Selective Scan) 在 TensorRT 上需要自定义 Plugin。
    *   探索 `Mamba.onnx` 导出方案。
2.  **端侧移植**:
    *   利用 Mamba 显存占用低的特性，尝试在 Jetson Orin 等边缘设备上部署。

---

## 执行建议 CheckList

- [ ] **Step 1**: 跑通 GOT-10k 测试，拿到第一个 AO 分数。
- [ ] **Step 2**: 观察显存占用，如果显存有富余，尝试增大 `NUMBER` (历史帧数) 到 10 甚至 20，看看精度收益。
- [ ] **Step 3**: (中长期) 开一个新的分支 `feature/stateful-inference`，开始啃这块硬骨头。
