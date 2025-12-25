import torch
import torch.nn as nn

class BaseBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_pretrained(self, checkpoint_path, model_name='vim'):
        """
        Load pretrained weights specifically for Vim or ViT.
        """
        print(f"Loading pretrained {model_name} from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # Handle pos_embed mismatch (interpolate)
        if 'pos_embed' in state_dict:
            # Logic to resize pos_embed if needed
            pass
            
        msg = self.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(msg.missing_keys)}")
        print(f"Unexpected keys: {len(msg.unexpected_keys)}")