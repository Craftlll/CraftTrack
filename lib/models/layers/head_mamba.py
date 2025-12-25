import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaSequenceHead(nn.Module):
    """
    用于 Mamba-ARTrack 的序列预测头。
    将 Mamba 的隐状态解码为坐标 Bin 的概率分布。
    """
    def __init__(self, hidden_dim, vocab_size=1000, num_layers=3):
        super().__init__()
        
        # 多层感知机 (MLP) 增强特征表达
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LayerNorm(hidden_dim))
            
        self.mlp = nn.Sequential(*layers)
        
        # 最终分类层：预测 [0, vocab_size-1] 的概率
        self.cls_head = nn.Linear(hidden_dim, vocab_size)
        
        # 坐标回归分支 (可选，用于辅助 Loss)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4) # 预测 sigmoid 归一化后的 [cx, cy, w, h]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: [B, D] or [B, L, D] (Mamba 的输出特征)
        """
        # 特征增强
        x = self.mlp(x)
        
        # 1. 分类 logits (用于 CrossEntropy Loss)
        logits = self.cls_head(x) # [B, Vocab]
        
        # 2. 连续坐标预测 (用于 L1/GIoU Loss)
        # 这里虽然是生成式，但在训练时加上回归辅助通常能加速收敛
        boxes = self.reg_head(x).sigmoid() # [B, 4] in [0,1]
        
        return logits, boxes