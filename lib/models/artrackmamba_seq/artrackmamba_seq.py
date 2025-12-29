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
        Matches ARTrackV2Seq forward signature and logic.
        """
        # Handle template input (ARTrackV2Seq uses template[:, 0])
        if template.dim() == 5:
            template = template[:, 0]
            
        # 1. Embeddings
        z_feat, x_feat = self.backbone.forward_embeddings(template, search)
        B = z_feat.shape[0]

        # 2. Construct Sequence (Aligned with ARTrackV2Seq)
        # Sequence: [Template, DynamicTemplate(opt), Search, History, Command]
        
        # 2.1 Command Tokens Construction (x0, y0, x1, y1, score)
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
             # seq_input contains indices of history trajectory
             trajectory = seq_input
             # Concat History + Command
             # ARTrackV2Seq: seqs_input_ = torch.cat([trajectory, command], dim=1)
             command = command.to(trajectory)
             seqs_input_ = torch.cat([trajectory, command], dim=1)
        else:
             # Fallback if no history provided (e.g. init), though ARTrackV2Seq usually has seq_input
             seqs_input_ = command

        seqs_input_ = seqs_input_.to(torch.int64).to(search.device)
        
        # 2.3 Embedding
        # Embed History+Command using shared word_embeddings
        tgt = self.word_embeddings(seqs_input_) # [B, L_seq, D]
        # In V2, tgt is permuted (1, 0, 2) in forward_features, but later transposed back?
        # V2: tgt = self.word_embeddings(seqs_input_).permute(1, 0, 2)
        # ... tgt += query_seq_embed[:, :tgt.shape[1]]
        # ... zxs = torch.cat((zx, tgt), dim=1) (Here zx is [B, L, D]?)
        # Actually V2's BaseBackbone uses [B, L, D] for concatenation.
        # Let's check V2 again: zxs = torch.cat((zx, tgt), dim=1). 
        # So tgt should be [B, L, D].
        
        # 2.4 Positional Embedding for Sequence
        query_command_embed_ = self.position_embeddings.weight.unsqueeze(0) # [1, 5, D]
        prev_embed_ = self.prev_position_embeddings.weight.unsqueeze(0) # [1, prenum*4, D]
        
        # Concatenate positional embeddings
        query_seq_embed = torch.cat([prev_embed_, query_command_embed_], dim=1) # [1, L_seq_total, D]
        query_seq_embed = query_seq_embed.repeat(B, 1, 1) # [B, L_seq_total, D]
        
        # Add Positional Embeddings
        # Ensure sizes match (handle cases where history might be partial or full)
        tgt = tgt + query_seq_embed[:, :tgt.shape[1], :]

        # 2.5 Prepare Dynamic Template (dz_feat)
        dz_tokens = None
        if dz_feat is not None:
            if dz_feat.dim() == 4:
                # Flatten [B, C, H, W] -> [B, H*W, C]
                dz_tokens = dz_feat.flatten(2).transpose(1, 2)
            else:
                dz_tokens = dz_feat

        # 2.6 Concatenate All Parts
        # Order: [Template(z_feat), DynamicTemplate(dz_tokens), Search(x_feat), History+Command(tgt)]
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
        # ARTrackV2Seq: 
        # x_out = self.norm(zxs[:, -5:-1]) (Box features)
        # score_feat = zxs[:, -1] (Score feature)
        
        x_out = self.norm(features[:, -5:-1, :])
        score_feat = features[:, -1, :] # [B, D]
        
        # NOTE: ARTrackV2Seq returns seq_feat = x_out (the box features before projection)
        seq_feat = x_out
        
        # Prediction using Weight Tying
        share_weight = self.word_embeddings.weight.T # [D, Vocab]
        possibility = torch.matmul(x_out, share_weight) # [B, 4, Vocab]
        pred_logits = possibility + self.output_bias
        
        # Apply Softmax to get probabilities (V2 does this and returns 'class')
        # out = out.softmax(-1)
        # But usually 'pred_logits' in DETR-like models refers to raw logits.
        # V2 returns {'class': values...} which are topk.
        # V2 returns {'feat': temp} where temp = out.transpose(0, 1). 
        # Wait, V2 `out` in line 206 is logits? 
        # out = possibility + self.output_bias
        # out_list.append(out.unsqueeze(0))
        # out = out.softmax(-1)
        # So 'feat' in V2 output seems to be the LOGITS (or close to it).
        # Let's return the logits as 'pred_logits'.
        
        # 5. Score Prediction
        score = self.score_mlp(score_feat)

        # 6. Construct Output Dictionary (Matching ARTrackV2Seq)
        out = {
            'pred_logits': pred_logits, # Corresponds to V2 'feat' (logits)
            # 'pred_boxes': ... # V2 doesn't return explicit boxes here, it returns logits/class indices.
            # But the Actor usually decodes them.
            # ARTrackMambaSeqActor expects 'pred_logits'.
            'score': score,
            'seq_feat': seq_feat, # [B, 4, D]
            'x_feat': x_feat.detach(), # V2 returns x_feat detached
        }
        
        # V2 also returns 'class' and 'seqs' (topk indices). 
        # If Actor needs them, we should compute them.
        # ARTrackMambaSeqActor computes loss from logits, so maybe not strictly needed for training?
        # But for inference/consistency:
        probs = pred_logits.softmax(-1)
        value, extra_seq = probs.topk(dim=-1, k=1)
        # extra_seq: [B, 4, 1] -> indices
        
        out['class'] = value.squeeze(-1) # [B, 4]
        out['seqs'] = extra_seq.squeeze(-1) # [B, 4]

        # 7. Template Update (Memory)
        loss = torch.tensor(0.0, dtype=torch.float32).to(search.device)
        if target_in_search_img is not None:
            # Logic from ARTrackV2Seq
            # Use dz_tokens if available, else z_feat? 
            # V2 uses z_1_feat (which corresponds to dz_tokens)
            curr_feat = dz_tokens if dz_tokens is not None else z_feat
            
            # Reshape to 2D for decoder: [B, H*W, C] -> [B, C, H, W]
            B_curr, L_curr, C_curr = curr_feat.shape
            H_curr = int(L_curr ** 0.5)
            curr_feat_2d = curr_feat.transpose(1, 2).reshape(B_curr, C_curr, H_curr, H_curr)
            
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
