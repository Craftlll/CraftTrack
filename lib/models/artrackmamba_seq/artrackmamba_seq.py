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
        # self.head = MambaSequenceHead(hidden_dim, self.vocab_size) # Removed to match ARTrackV2Seq
        
        # 3. Embeddings for Inputs
        self.bins = cfg.MODEL.BINS
        self.range = cfg.MODEL.RANGE
        self.prenum = cfg.MODEL.PRENUM
        # ARTrackV2Seq uses specific size: bins * range + 6
        self.word_embeddings = nn.Embedding(self.bins * self.range + 6, hidden_dim, padding_idx=self.bins * self.range+4, max_norm=1, norm_type=2.0)
        
        # Command Tokens (for box prediction) - ARTrackV2Seq doesn't use learned cmd_tokens but embeds indices
        # self.cmd_tokens = nn.Parameter(torch.zeros(1, 4, hidden_dim)) 
        # nn.init.normal_(self.cmd_tokens, std=0.02)
        
        # CLS Token (for score prediction) - Removed to match ARTrackV2Seq sequence structure
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        # nn.init.normal_(self.cls_token, std=0.02)

        # Positional Embeddings matching ARTrackV2Seq
        self.position_embeddings = nn.Embedding(5, hidden_dim)
        self.output_bias = torch.nn.Parameter(torch.zeros(self.bins * self.range + 6))
        self.prev_position_embeddings = nn.Embedding(self.prenum * 4, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim) # For output projection

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
        Matches ARTrackV2Seq forward signature and logic with Vim backbone.
        """
        # Handle template input (ARTrackV2Seq uses template[:, 0])
        if template.dim() == 5:
            template = template[:, 0]
            
        # 1. Embeddings (Initial Template & Search)
        # z_feat/x_feat ALREADY have pos_embeddings added inside forward_embeddings
        z_feat, x_feat = self.backbone.forward_embeddings(template, search)
        B = z_feat.shape[0]

        # 2. Construct Sequence parts
        
        # 2.1 Command Tokens Construction
        x0 = self.bins * self.range
        y0 = self.bins * self.range + 1
        x1 = self.bins * self.range + 2
        y1 = self.bins * self.range + 3
        score_idx = self.bins * self.range + 5

        command = torch.cat([
            torch.ones((B, 1)).to(search.device) * x0,
            torch.ones((B, 1)).to(search.device) * y0,
            torch.ones((B, 1)).to(search.device) * x1,
            torch.ones((B, 1)).to(search.device) * y1,
            torch.ones((B, 1)).to(search.device) * score_idx
        ], dim=1)
        
        # 2.2 History/Prompt Construction
        if seq_input is not None:
             trajectory = seq_input
             command = command.to(trajectory)
             seqs_input_ = torch.cat([trajectory, command], dim=1)
        else:
             seqs_input_ = command

        seqs_input_ = seqs_input_.to(torch.int64).to(search.device)
        
        # 2.3 Embedding for History/Command
        tgt = self.word_embeddings(seqs_input_) # [B, L_seq, D]
        
        # 2.4 Positional Embedding for Sequence (History + Command)
        query_command_embed_ = self.position_embeddings.weight.unsqueeze(0) # [1, 5, D]
        prev_embed_ = self.prev_position_embeddings.weight.unsqueeze(0) # [1, prenum*4, D]
        
        query_seq_embed = torch.cat([prev_embed_, query_command_embed_], dim=1)
        query_seq_embed = query_seq_embed.repeat(B, 1, 1) # [B, L_seq_total, D]
        
        # Add Positional Embeddings to tgt
        tgt = tgt + query_seq_embed[:, :tgt.shape[1], :]

        # 2.5 Prepare Dynamic Template (dz_feat)
        dz_tokens = None
        if dz_feat is not None:
            if dz_feat.dim() == 4:
                # Flatten [B, C, H, W] -> [B, H*W, C] -> [B, L, D]
                dz_tokens = dz_feat.flatten(2).transpose(1, 2)
            else:
                dz_tokens = dz_feat
            
            # === 修正开始: 为 Dynamic Template 添加位置编码 ===
            # dz_tokens 通常和 z_feat 具有相同的空间分辨率
            # 我们可以直接复用 backbone 中的位置编码生成逻辑
            if hasattr(self.backbone, 'get_posembed'):
                dz_pos_embed = self.backbone.get_posembed(dz_tokens.shape[1])
                dz_tokens = dz_tokens + dz_pos_embed
            # === 修正结束 ===

        # 2.6 Concatenate All Parts
        # Order: [Template(z), Dynamic(dz), Search(x), History+Command(tgt)]
        # This order allows Causal Mamba to let Search see Template, and Command see everything.
        parts = [z_feat]
        if dz_tokens is not None:
            parts.append(dz_tokens)
        parts.append(x_feat)
        parts.append(tgt)
        
        seq = torch.cat(parts, dim=1)

        # 3. Mamba Backbone
        # Pass the full sequence through Mamba
        features = self.backbone.forward_backbone(seq)

        # 4. Extract Outputs (Weight Tying Prediction)
        # Extract features corresponding to command tokens (last 5 tokens: x0, y0, x1, y1, score)
        x_out = self.norm(features[:, -5:-1, :]) # Box tokens
        score_feat = features[:, -1, :]          # Score token
        
        # NOTE: ARTrackV2Seq returns seq_feat = x_out
        seq_feat = x_out
        
        # Prediction using Weight Tying
        share_weight = self.word_embeddings.weight.T # [D, Vocab]
        possibility = torch.matmul(x_out, share_weight) # [B, 4, Vocab]
        pred_logits = possibility + self.output_bias
        
        # 5. Score Prediction
        score = self.score_mlp(score_feat)

        # 6. Construct Output Dictionary
        out = {
            'pred_logits': pred_logits,
            'score': score,
            'seq_feat': seq_feat,
            'x_feat': x_feat.detach(),
        }
        
        # Optional: Compute 'class' and 'seqs' for inference convenience
        probs = pred_logits.softmax(-1)
        value, extra_seq = probs.topk(dim=-1, k=1)
        out['class'] = value.squeeze(-1)
        out['seqs'] = extra_seq.squeeze(-1)

        # 7. Template Update Logic (Same as V2)
        loss = torch.tensor(0.0, dtype=torch.float32).to(search.device)
        if target_in_search_img is not None:
            # Training update logic
            # Use z_feat as fallback if dz_tokens implies initial state
            # Note: dz_tokens here has pos_embed added, but for decoder we might want raw features?
            # V2 passes z_1_feat. Usually Decoder handles it. 
            # If dz_tokens was modified in-place above, use caution. My code above modifies `dz_tokens` variable, not `dz_feat`.
            curr_feat = dz_tokens if dz_tokens is not None else z_feat
            
            B_curr, L_curr, C_curr = curr_feat.shape
            H_curr = int(L_curr ** 0.5)
            # Reshape back to 2D for Map Decoder
            curr_feat_2d = curr_feat.transpose(1, 2).reshape(B_curr, C_curr, H_curr, H_curr)
            
            target_in_search_gt = target_in_search_img
            
            update_img, loss_temp = self.cross_2_decoder(curr_feat_2d, target_in_search_gt)
            update_feat = self.cross_2_decoder.patchify(update_img)
            
            out['dz_feat'] = update_feat
            loss += loss_temp
            out['renew_loss'] = loss
            
        else:
            # Inference update logic
            if dz_tokens is not None:
                curr_feat = dz_tokens
                B_curr, L_curr, C_curr = curr_feat.shape
                H_curr = int(L_curr ** 0.5)
                curr_feat_2d = curr_feat.transpose(1, 2).reshape(B_curr, C_curr, H_curr, H_curr)
                
                update_feat = self.cross_2_decoder(curr_feat_2d, eval=True)
                update_feat = self.cross_2_decoder.patchify(update_feat)
                out['dz_feat'] = update_feat
            else:
                out['dz_feat'] = None

        return out

def build_artrackmamba_seq(cfg, training=True):
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

    # 4. Load Pretrained Weights (Consistent with ARTrackV2Seq)
    load_from = getattr(cfg.MODEL, "PRETRAIN_PTH", None)
    if load_from is not None and load_from != '':
        print("Loading ARTrackMambaSeq from", load_from)
        checkpoint = torch.load(load_from, map_location="cpu")
        if 'net' in checkpoint:
             missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        elif 'model' in checkpoint:
             missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
        else:
             missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        print('Load pretrained model from: ' + load_from)
    else:
        print("Warning: No pretrained weights loaded for ARTrackMambaSeq (Benchmarking random init)")

    # Optional: Load specific backbone/sequence weights if specified (Legacy/Specialized)
    pretrain_file = getattr(cfg.MODEL, "PRETRAIN_FILE", "")
    if 'sequence' in pretrain_file and training:
        print("Loading sequence pretrained weights...")
        checkpoint = torch.load(pretrain_file, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + pretrain_file)
    
    return model
