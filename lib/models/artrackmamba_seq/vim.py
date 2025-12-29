import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from lib.models.layers.patch_embed_mamba import PatchEmbed
from lib.utils.misc import is_main_process

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: mamba_ssm not found.")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class VimBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        self.norm = norm_cls(dim)
        
        # 参数量优化：共享权重的双向 Mamba
        self.mixer = mixer_cls(dim)
        
        # [回退] 移除 MLP Gate，使用简单融合以排查梯度爆炸
        # self.gate_fc = ... 
        
        # [改进] Local Convolution (Inductive Bias) - 保留，因为它对视觉任务很重要且通常比较稳定
        # Mamba 擅长 Global，引入一个 depthwise conv 补充 Local 归纳偏置
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        
        # [回退] 移除 LayerScale，使用标准 Residual
        # self.gamma = ...
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Init local conv to be identity-like
        nn.init.constant_(self.local_conv.weight, 1.0/3.0) # Average smoothing init
        nn.init.constant_(self.local_conv.bias, 0)

    def forward(self, x, inference_params=None):
        # x: [B, L, D]
        shortcut = x
        x = self.norm(x)
        
        # [改进] Local Feature Extraction
        # 转置为 [B, D, L] 进行卷积，再转回来
        x_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_local # Residual connection for local conv
        
        # Bidirectional Scanning
        # Forward
        x_fwd = self.mixer(x)
        
        # Backward (Reuse the same mixer)
        x_rev = x.flip([1])
        x_bwd = self.mixer(x_rev).flip([1])
        
        # [回退] Simple Mean Fusion
        x_out = (x_fwd + x_bwd) / 2.0
        
        # [回退] Standard DropPath
        x = shortcut + self.drop_path(x_out)
        return x

class VisionMamba(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 stride=16,
                 in_chans=3, 
                 embed_dim=192, 
                 depth=24, 
                 ssm_d_state=16, 
                 ssm_dt_rank="auto",
                 ssm_ratio=2.0,
                 ssm_conv=3,
                 ssm_conv_bias=True,
                 ssm_forward_type="v2", 
                 mlp_ratio=4., 
                 drop_rate=0., 
                 drop_path_rate=0.1, 
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, 
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. Tokens & Pos Embed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # [改进] Search Region Pos Bias
        # 显式可学习偏差，用于区分 Template 和 Search 区域的位置空间
        self.search_pos_bias = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.layers = nn.ModuleList([
            VimBlock(
                dim=embed_dim, 
                mixer_cls=partial(Mamba, d_state=ssm_d_state, d_conv=ssm_conv, expand=ssm_ratio),
                norm_cls=norm_layer,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # 4. Final Norm
        self.norm = norm_layer(embed_dim)
        
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        trunc_normal_(self.search_pos_bias, std=.02) # 初始化 Bias
        
        self.apply(self._init_weights_modules)

    def _init_weights_modules(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                if not getattr(m.bias, "_no_reinit", False):
                     nn.init.constant_(m.bias, 0)
                     
        elif isinstance(m, nn.LayerNorm) or isinstance(m, RMSNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        # Mamba 稳健初始化
        if isinstance(m, Mamba):
            if hasattr(m, 'out_proj'):
                nn.init.constant_(m.out_proj.weight, 0)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)

    def get_posembed(self, seq_len, start_index=1):
        pos_embed_no_cls = self.pos_embed[:, 1:, :] 
        if seq_len == pos_embed_no_cls.shape[1]:
            return pos_embed_no_cls
        
        N = pos_embed_no_cls.shape[1]
        
        # 处理非正方形插值
        if int(math.sqrt(seq_len)) ** 2 != seq_len:
             pos_embed_no_cls = pos_embed_no_cls.transpose(1, 2)
             pos_embed_no_cls = F.interpolate(pos_embed_no_cls, size=(seq_len), mode='linear', align_corners=False)
             pos_embed_no_cls = pos_embed_no_cls.transpose(1, 2)
             return pos_embed_no_cls
             
        orig_size = int(math.sqrt(N))
        new_size = int(math.sqrt(seq_len))
        
        pos_embed_no_cls = pos_embed_no_cls.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_embed_no_cls = F.interpolate(pos_embed_no_cls, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_embed_no_cls = pos_embed_no_cls.flatten(2).transpose(1, 2)
        return pos_embed_no_cls

    def forward_embeddings(self, z, x):
        """
        仅执行 Embedding 和 Positional Encoding，不进入 Layers。
        """
        # Patch Embed
        z_feat = self.patch_embed(z)
        x_feat = self.patch_embed(x)
        
        # Pos Embed
        # Template 使用原始 Pos Embed
        z_feat = z_feat + self.get_posembed(z_feat.shape[1])
        
        # Search 使用 Pos Embed + 额外的 Learnable Bias
        # 这有助于模型区分“这是模板”还是“这是搜索区域”，避免位置混淆
        x_feat = x_feat + self.get_posembed(x_feat.shape[1]) + self.search_pos_bias
        
        return z_feat, x_feat

    def forward_backbone(self, x_cat):
        x_cat = self.pos_drop(x_cat)
        for layer in self.layers:
            x_cat = layer(x_cat)
        x_cat = self.norm(x_cat)
        return x_cat

    def forward(self, z, x, **kwargs):
        z_feat, x_feat = self.forward_embeddings(z, x)
        x_cat = torch.cat([z_feat, x_feat], dim=1)
        return self.forward_backbone(x_cat)

    def load_pretrained(self, checkpoint_path):
        if is_main_process():
            print(f"Loading Vim pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k] = v

        # 兼容性处理
        for k in list(new_state_dict.keys()):
            if "conv1d.weight" in k:
                weight = new_state_dict[k]
                if weight.shape[-1] != 3: 
                     new_state_dict[k] = weight[..., :3]

        keys_to_ignore = ['head.weight', 'head.bias', 'search_pos_bias']
        for k in keys_to_ignore:
            if k in new_state_dict: del new_state_dict[k]
                
        msg = self.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            print(f"Loaded successfully. Missing keys: {len(msg.missing_keys)}")

# Builders
def vim_tiny_patch16_224_bimamba_v2_final(**kwargs):
    # 使用 RMSNorm 可能会更加稳定，这里保留 LayerNorm 作为默认，但可以通过 kwargs 覆盖
    return VisionMamba(embed_dim=192, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)

def vim_small_patch16_224_bimamba_v2_final(**kwargs):
    return VisionMamba(embed_dim=384, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)

def vim_base_patch16_224_bimamba_v2_final(**kwargs):
    return VisionMamba(embed_dim=768, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)