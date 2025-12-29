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
        
        # [优化关键点] 参数量优化：共享权重的双向 Mamba
        # 之前使用了两个独立的 mixer 导致参数量翻倍。现在改为共享同一个 mixer。
        # 虽然计算量(FLOPs)依然是 2 倍，但参数量减半，内存带宽压力减小。
        self.mixer = mixer_cls(dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, inference_params=None):
        # x: [B, L, D]
        shortcut = x
        x = self.norm(x)
        
        # Bidirectional Scanning with Shared Weights
        # 1. Forward Pass
        x_fwd = self.mixer(x)
        
        # 2. Backward Pass (Reuse the same mixer)
        # Flip input -> Run Mixer -> Flip output back
        x_rev = x.flip([1])
        x_bwd = self.mixer(x_rev).flip([1])
        
        # Fusion
        x_out = (x_fwd + x_bwd) / 2.0
        
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
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 3. Layers
        # 逐渐增加 drop path rate
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
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_modules)

    def _init_weights_modules(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # 保护 Mamba 内部参数初始化
                if not getattr(m.bias, "_no_reinit", False):
                     nn.init.constant_(m.bias, 0)
                     
        elif isinstance(m, nn.LayerNorm) or isinstance(m, RMSNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
        # Mamba 稳健初始化
        # 初始化 Mamba 输出投影层为 0，使得初始状态下相当于 Identity Mapping
        # 这有助于深层网络的训练稳定性
        if isinstance(m, Mamba):
            if hasattr(m, 'out_proj'):
                nn.init.constant_(m.out_proj.weight, 0)
                if m.out_proj.bias is not None:
                    nn.init.constant_(m.out_proj.bias, 0)

    def get_posembed(self, seq_len, start_index=1):
        # 动态位置编码插值，用于适配不同分辨率输入
        pos_embed_no_cls = self.pos_embed[:, 1:, :] 
        if seq_len == pos_embed_no_cls.shape[1]:
            return pos_embed_no_cls
        
        N = pos_embed_no_cls.shape[1]
        
        # 处理非正方形的情况，假设 seq_len 是 H*W
        # 如果无法开方整数，则做简单插值
        if int(math.sqrt(seq_len)) ** 2 != seq_len:
             # 如果不是正方形，尝试作为 1D 序列插值
             pos_embed_no_cls = pos_embed_no_cls.transpose(1, 2) # [1, D, N]
             pos_embed_no_cls = F.interpolate(pos_embed_no_cls, size=(seq_len), mode='linear', align_corners=False)
             pos_embed_no_cls = pos_embed_no_cls.transpose(1, 2) # [1, seq_len, D]
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
        用于在主模型中插入 Prompt Tokens。
        """
        # Patch Embed
        z_feat = self.patch_embed(z)
        x_feat = self.patch_embed(x)
        
        # Pos Embed
        z_feat = z_feat + self.get_posembed(z_feat.shape[1])
        x_feat = x_feat + self.get_posembed(x_feat.shape[1])
        
        return z_feat, x_feat

    def forward_backbone(self, x_cat):
        """
        接收拼接好的序列 (Template + Prompts + Search)，执行 SSM 建模。
        """
        x_cat = self.pos_drop(x_cat)
        
        for layer in self.layers:
            x_cat = layer(x_cat)
            
        x_cat = self.norm(x_cat)
        return x_cat

    def forward(self, z, x, **kwargs):
        # 兼容旧接口：直接处理 Z 和 X
        z_feat, x_feat = self.forward_embeddings(z, x)
        x_cat = torch.cat([z_feat, x_feat], dim=1)
        return self.forward_backbone(x_cat)

    def load_pretrained(self, checkpoint_path):
        if is_main_process():
            print(f"Loading Vim pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # [修改] 恢复常规加载逻辑，不再需要处理 fwd/bwd 映射
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k] = v

        # 兼容性处理：conv1d 权重尺寸及其他过滤
        for k in list(new_state_dict.keys()):
            if "conv1d.weight" in k:
                weight = new_state_dict[k]
                # 假设当前模型 conv 是 3，预训练是 4
                if weight.shape[-1] != 3: 
                     # 简单的 slice，具体取决于预训练实现
                     new_state_dict[k] = weight[..., :3]

        keys_to_ignore = ['head.weight', 'head.bias']
        for k in keys_to_ignore:
            if k in new_state_dict: del new_state_dict[k]
                
        msg = self.load_state_dict(new_state_dict, strict=False)
        if is_main_process():
            print(f"Loaded successfully. Missing keys: {len(msg.missing_keys)}")

# Builders
# 保持 ssm_ratio=2.0 也是控制参数量的关键 (Vim默认是2.0, Mamba论文是2.0)
def vim_tiny_patch16_224_bimamba_v2_final(**kwargs):
    # 使用 RMSNorm 可能会更加稳定，这里保留 LayerNorm 作为默认，但可以通过 kwargs 覆盖
    return VisionMamba(embed_dim=192, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)

def vim_small_patch16_224_bimamba_v2_final(**kwargs):
    return VisionMamba(embed_dim=384, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)

def vim_base_patch16_224_bimamba_v2_final(**kwargs):
    return VisionMamba(embed_dim=768, depth=24, ssm_d_state=16, ssm_ratio=2.0, **kwargs)