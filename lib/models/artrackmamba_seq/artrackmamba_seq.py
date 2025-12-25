import torch
import torch.nn as nn
from lib.models.artrackmamba_seq.base_backbone import BaseBackbone
from lib.models.artrackmamba_seq import vim as vim_models
from lib.models.layers.head_mamba import MambaSequenceHead

# 假设你已经有了 MemoryBank 的实现（参考之前的 MemoryBankModule 代码）
# from lib.models.modules.memory_bank import MemoryBankModule 

class ARTrackMambaSeq(BaseBackbone):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        
        # 1. Backbone (Vim)
        self.backbone = getattr(vim_models, cfg.MODEL.BACKBONE.TYPE)(
            stride=cfg.MODEL.BACKBONE.STRIDE,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        if cfg.MODEL.BACKBONE.PRETRAINED:
            self.backbone.load_pretrained(cfg.MODEL.BACKBONE.PRETRAINED_PATH)
            
        hidden_dim = self.backbone.embed_dim
        
        # 2. Heads
        self.vocab_size = cfg.MODEL.HEAD.VOCAB_SIZE
        self.head = MambaSequenceHead(hidden_dim, self.vocab_size)
        
        # 3. Embeddings for Inputs
        # 将离散的坐标词 ([0~999]) 映射回向量，用于作为输入喂给 Mamba
        self.word_embeddings = nn.Embedding(self.vocab_size, hidden_dim)
        
        # Command Tokens (指示模型该输出什么)
        # 例如: <start_x>, <start_y>, <start_w>, <start_h>
        self.cmd_tokens = nn.Parameter(torch.zeros(1, 4, hidden_dim)) 
        nn.init.normal_(self.cmd_tokens, std=0.02)
        
        # 4. Memory Bank (RAG)
        if cfg.MODEL.MEMORY.USE:
            # 初始化记忆库模块 (需自行实现或简化)
            # self.memory_bank = MemoryBankModule(...)
            pass

    def forward(self, template, search, seq_input=None, gt_bboxes=None):
        """
        template: [B, 3, 128, 128]
        search:   [B, 3, 256, 256]
        seq_input: [B, L] 历史轨迹的离散坐标索引 (Training only)
        """
        # 1. 图像特征提取 (Patch Embed + Pos Embed)
        z_feat, x_feat = self.backbone.forward_embeddings(template, search)
        B = z_feat.shape[0]

        # 2. 构造输入序列
        # -----------------------------------------------------------
        # ARTrackV2 逻辑: [Template, History_Prompt, Search, Command]
        # Mamba 是自回归的，我们需要把 "History" 放在 "Command" 之前
        
        if self.training:
            # 训练模式 (Teacher Forcing)
            # seq_input 是 GT 的坐标 token，我们把它嵌入后作为 Prompt
            # seq_input shape: [B, 4] (x,y,w,h tokens)
            
            # 将 GT tokens 变为向量
            # 注意：实际 ARTrack 训练时可能并不是逐 Token 预测，而是基于 Search 预测
            # 这里简化为：Search 区域作为 Context，Command Token 触发预测
            
            cmd_tokens = self.cmd_tokens.expand(B, -1, -1)
            
            # 序列拼接: [Template (64), Search (256), Command (4)]
            # Vim 是双向的，所以顺序不敏感，但逻辑上 Command 在最后收集信息
            seq = torch.cat([z_feat, x_feat, cmd_tokens], dim=1)
            
            # 3. Mamba 混合 (Mixing)
            features = self.backbone.forward_backbone(seq)
            
            # 提取 Command Token 对应的输出特征
            # Command tokens 是最后 4 个
            out_feat = features[:, -4:, :] # [B, 4, D]
            
            # 4. 预测
            pred_logits, pred_boxes = self.head(out_feat)
            
            # 5. 构造输出字典 (匹配 ltr_trainer)
            out = {
                'pred_logits': pred_logits, # [B, 4, Vocab]
                'pred_boxes': pred_boxes    # [B, 4, 4] (如果要做额外的 box regression)
            }
            return out
            
        else:
            # 推理模式 (Inference)
            # 只需要通过一次前向传播，利用 Vim 的全局感受野预测坐标
            cmd_tokens = self.cmd_tokens.expand(B, -1, -1)
            seq = torch.cat([z_feat, x_feat, cmd_tokens], dim=1)
            
            features = self.backbone.forward_backbone(seq)
            out_feat = features[:, -4:, :]
            
            pred_logits, pred_boxes = self.head(out_feat)
            return pred_logits, pred_boxes

def build_artrack_mamba_seq(cfg):
    return ARTrackMambaSeq(cfg)