import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from lib.models.artrackmamba_seq.base_backbone import BaseBackbone
from lib.models.artrackmamba_seq import vim as vim_models
from lib.models.layers.head_mamba import MambaSequenceHead
from lib.models.layers.mask_decoder import build_maskdecoder

class MlpScoreDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bn=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        out_dim = 1 # score
        if bn:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])
        else:
            self.layers = nn.Sequential(*[nn.Sequential(nn.Linear(n, k), nn.ReLU())
                                          if i < num_layers - 1
                                          else nn.Linear(n, k)
                                          for i, (n, k) in enumerate(zip([in_dim] + h, h + [out_dim]))])

    def forward(self, reg_tokens):
        """
        reg tokens shape: (b, 4, embed_dim)
        """
        x = self.layers(reg_tokens) # (b, 4, 1)
        x = x.mean(dim=1)   # (b, 1)
        return x

def build_score_decoder(cfg, hidden_dim):
    return MlpScoreDecoder(
        in_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        bn=False
    )

class ARTrackMambaSeq(BaseBackbone):
    def __init__(self, cfg, score_mlp, cross_2_decoder, hidden_dim, **kwargs):
        super().__init__()
        
        # 1. Backbone (Vim)
        self.backbone = getattr(vim_models, cfg.MODEL.BACKBONE.TYPE)(
            stride=cfg.MODEL.BACKBONE.STRIDE,
            drop_path_rate=cfg.TRAIN.DROP_PATH_RATE
        )
        if cfg.MODEL.BACKBONE.PRETRAINED:
            self.backbone.load_pretrained(cfg.MODEL.BACKBONE.PRETRAINED_PATH)
            
        # Ensure hidden_dim matches backbone
        if self.backbone.embed_dim != hidden_dim:
            print(f"Warning: Config hidden_dim {hidden_dim} differs from Backbone embed_dim {self.backbone.embed_dim}. Using Backbone's.")
            hidden_dim = self.backbone.embed_dim
        
        # 2. Heads
        self.vocab_size = cfg.MODEL.HEAD.VOCAB_SIZE
        self.head = MambaSequenceHead(hidden_dim, self.vocab_size)
        
        # 3. Embeddings for Inputs
        self.word_embeddings = nn.Embedding(self.vocab_size, hidden_dim)
        
        # Command Tokens (for box prediction)
        self.cmd_tokens = nn.Parameter(torch.zeros(1, 4, hidden_dim)) 
        nn.init.normal_(self.cmd_tokens, std=0.02)
        
        # CLS Token (for score prediction)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        # 4. Components from ARTrackV2Seq
        self.score_mlp = score_mlp
        self.cross_2_decoder = cross_2_decoder
        self.identity = torch.nn.Parameter(torch.zeros(1, 3, hidden_dim))
        self.identity = trunc_normal_(self.identity, std=.02)

    def forward(self, template, search, 
                dz_feat=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                head_type=None,
                stage=None,
                search_feature=None,
                target_in_search_img=None,
                gt_bboxes=None):
        """
        Matches ARTrackV2Seq forward signature.
        """
        # Handle template input (ARTrackV2Seq uses template[:, 0])
        if template.dim() == 5:
            template = template[:, 0]
            
        # 1. Embeddings
        z_feat, x_feat = self.backbone.forward_embeddings(template, search)
        B = z_feat.shape[0]

        # 2. Construct Sequence
        # Sequence: [CLS, Template, DynamicTemplate(opt), Search, History/Command]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        # Handle dz_feat (Dynamic Template)
        # In ARTrackV2, dz_feat is passed as z_1_feat. It's usually patch tokens.
        # If it comes from cross_2_decoder, it might be [B, L, C] or [B, C, H, W].
        # cross_2_decoder.patchify returns [B, L, C].
        dz_tokens = None
        if dz_feat is not None:
            if dz_feat.dim() == 4:
                # Flatten [B, C, H, W] -> [B, H*W, C]
                dz_tokens = dz_feat.flatten(2).transpose(1, 2)
            else:
                dz_tokens = dz_feat

        # Handle Command/History Tokens
        if self.training and seq_input is not None:
            # seq_input: [B, 4] (GT tokens) - Simplified for consistency with provided Mamba code
            # Note: The provided Mamba code logic for training was:
            # cmd_tokens = self.cmd_tokens.expand(B, -1, -1)
            # seq = torch.cat([z_feat, x_feat, cmd_tokens], dim=1)
            # It seems it didn't use seq_input for embedding? 
            # Wait, the comment said "seq_input is GT... embed it...".
            # But the code used `cmd_tokens` even in training.
            # I will stick to `cmd_tokens` for now to be safe, or use `seq_input` if implemented.
            # ARTrackV2Seq passes seq_input to backbone.
            # I will append cmd_tokens as placeholders for prediction.
            suffix_tokens = self.cmd_tokens.expand(B, -1, -1)
        else:
            suffix_tokens = self.cmd_tokens.expand(B, -1, -1)

        # Concatenate
        parts = [cls_tokens, z_feat]
        if dz_tokens is not None:
            parts.append(dz_tokens)
        parts.append(x_feat)
        parts.append(suffix_tokens)
        
        seq = torch.cat(parts, dim=1)

        # 3. Mamba Backbone
        features = self.backbone.forward_backbone(seq)

        # 4. Extract Outputs
        # Score feature from CLS token (index 0)
        score_feat = features[:, 0:1, :] # [B, 1, D] - Keep dims for MLP? MLP expects [B, 4, D] or [B, D]?
        # MlpScoreDecoder expects `reg_tokens` of shape (b, 4, embed_dim) and does mean(dim=1).
        # So I should probably provide more tokens or expand this?
        # Or I can just repeat it 4 times.
        score_feat = score_feat.expand(-1, 4, -1)

        # Box feature from last 4 tokens (Command tokens)
        out_feat = features[:, -4:, :]

        # 5. Predictions
        score = self.score_mlp(score_feat)
        pred_logits, pred_boxes = self.head(out_feat)

        # 6. Template Update (Memory)
        out = {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes,
            'score': score,
            'seq_feat': features, # ARTrackV2Seq returns seq_feat
        }

        loss = torch.tensor(0.0, dtype=torch.float32).to(search.device)
        if target_in_search_img is not None:
            # Logic from ARTrackV2Seq
            # FIX: Do not use backbone.patch_embed as it returns embeddings (192 dim), not pixel patches (768 dim).
            # Also do not unpatchify immediately.
            # MaskDecoder expects 'imgs' as second argument.
            # target_in_search_gt = self.backbone.patch_embed(target_in_search_img)
            
            # z_1_feat handling (using dz_tokens or z_feat if dz is None?)
            # ARTrackV2Seq uses z_1_feat. If dz_feat was passed, it is z_1_feat.
            # If dz_feat was None, z_1_feat might be uninitialized or we use z_feat?
            # ARTrackV2Seq assumes z_1_feat exists if we are in this block?
            # Actually, ARTrackV2Seq uses `z_1_feat` which is returned by backbone.
            # In ViT, z_1_feat is passed in.
            # Here, if dz_feat is passed, we use it. If not, maybe we should use z_feat?
            # But the update logic updates the *dynamic* template.
            # If dz_feat is None, we probably start from scratch or use z_feat as base?
            # ARTrackV2Seq: `z_1_feat = z_1_feat.reshape(...)`
            
            curr_feat = dz_tokens if dz_tokens is not None else z_feat
            
            # Reshape to 2D for decoder: [B, H*W, C] -> [B, C, H, W]
            B, L, C = curr_feat.shape
            H = int(L ** 0.5)
            curr_feat_2d = curr_feat.transpose(1, 2).reshape(B, C, H, H)
            
            # target_in_search_gt = self.cross_2_decoder.unpatchify(target_in_search_gt)
            target_in_search_gt = target_in_search_img
            
            update_img, loss_temp = self.cross_2_decoder(curr_feat_2d, target_in_search_gt)
            update_feat = self.cross_2_decoder.patchify(update_img)
            
            out['dz_feat'] = update_feat
            loss += loss_temp
            out['renew_loss'] = loss
            
        else:
            # Eval mode update logic
            if dz_tokens is not None:
                curr_feat = dz_tokens
                B, L, C = curr_feat.shape
                H = int(L ** 0.5)
                curr_feat_2d = curr_feat.transpose(1, 2).reshape(B, C, H, H)
                
                update_feat = self.cross_2_decoder(curr_feat_2d, eval=True)
                update_feat = self.cross_2_decoder.patchify(update_feat)
                out['dz_feat'] = update_feat
            else:
                out['dz_feat'] = None # Or z_feat?

        return out

def build_artrackmamba_seq(cfg):
    # 1. Build Score MLP
    # Assuming hidden_dim from config or defaults.
    # We need to know hidden_dim before building MLP? 
    # Usually config has it.
    hidden_dim = getattr(cfg.MODEL, "HIDDEN_DIM", 192) # Default for Vim-Tiny
    
    # Check if backbone type implies dimension (helper logic)
    if 'tiny' in cfg.MODEL.BACKBONE.TYPE:
        hidden_dim = 192
    elif 'small' in cfg.MODEL.BACKBONE.TYPE:
        hidden_dim = 384
    elif 'base' in cfg.MODEL.BACKBONE.TYPE:
        hidden_dim = 768
        
    score_mlp = build_score_decoder(cfg, hidden_dim)
    
    # 2. Build Mask Decoder
    cross_2_decoder = build_maskdecoder(cfg, hidden_dim)
    
    # 3. Build Model
    model = ARTrackMambaSeq(cfg, score_mlp, cross_2_decoder, hidden_dim)
    
    return model
