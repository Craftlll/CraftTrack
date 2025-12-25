import os
import sys
import torch
import time
from easydict import EasyDict as edict

# ==========================================
# 1. 智能路径设置 (兼容 Root 运行和子目录运行)
# ==========================================
current_file_path = os.path.abspath(__file__)
project_root = ""

# 如果脚本在 lib/models/... 下，回溯 3 级；如果在根目录，直接使用当前目录
if "lib/models" in current_file_path:
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_file_path), "../../../"))
else:
    project_root = os.path.dirname(current_file_path)

if project_root not in sys.path:
    print(f"[ENV] Adding project root to path: {project_root}")
    sys.path.append(project_root)

# ==========================================
# 2. 导入模块
# ==========================================
try:
    from lib.models.artrackmamba_seq.artrackmamba_seq import build_artrack_mamba_seq
    # 尝试预加载一下 mamba，防止模型构建时才报底层的错
    import mamba_ssm
    print("[INFO] Successfully imported modules and mamba_ssm.")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print(">>> 提示: 请确保安装了 'mamba_ssm' 且在正确的目录下运行。")
    print(">>> 运行目录应为项目根目录 (CraftTrack/) 或脚本所在目录。")
    sys.exit(1)

def get_dummy_config():
    """配置模拟"""
    cfg = edict()
    cfg.MODEL = edict()
    cfg.MODEL.BACKBONE = edict()
    # 对应 vim.py 中的函数名
    cfg.MODEL.BACKBONE.TYPE = "vim_tiny_patch16_224_bimamba_v2_final" 
    cfg.MODEL.BACKBONE.STRIDE = 16
    cfg.MODEL.BACKBONE.PRETRAINED = False 
    cfg.MODEL.BACKBONE.PRETRAINED_PATH = ""
    
    cfg.MODEL.HEAD = edict()
    cfg.MODEL.HEAD.VOCAB_SIZE = 1000 
    
    # 增加 Memory 配置防止 KeyError
    cfg.MODEL.MEMORY = edict()
    cfg.MODEL.MEMORY.USE = False 
    
    cfg.TRAIN = edict()
    cfg.TRAIN.DROP_PATH_RATE = 0.1
    return cfg

def test_inference():
    # -----------------------------------------------
    # 强制检查 CUDA (Mamba 核心算子必须用 GPU)
    # -----------------------------------------------
    if not torch.cuda.is_available():
        print("[FATAL] CUDA is not available. Mamba requires a GPU to run.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"[INFO] Testing on device: {torch.cuda.get_device_name(0)}")

    # 1. Build
    print("-" * 30)
    print("[INFO] Building ARTrackMambaSeq model...")
    cfg = get_dummy_config()
    
    try:
        model = build_artrack_mamba_seq(cfg)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Stats
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Build Success. Trainable Params: {param_count / 1e6:.2f} M")
    
    # Check Class Name
    backbone_name = model.backbone.__class__.__name__
    if 'VisionMamba' in backbone_name:
        print(f"[CHECK] Backbone Class Verification: PASS ({backbone_name})")
    else:
        print(f"[CHECK] Backbone Class Verification: WARNING (Got {backbone_name})")

    # 3. Dummy Input (Simulate ARTrackV2 Shapes)
    batch_size = 2
    # Template: 128x128
    template = torch.randn(batch_size, 3, 128, 128).to(device)
    # Search: 256x256
    search = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print("-" * 30)
    print(f"[INFO] Input Template: {template.shape}")
    print(f"[INFO] Input Search:   {search.shape}")

    # 4. Forward Pass
    print("[INFO] Running warm-up pass...")
    try:
        with torch.no_grad():
            # Warmup
            _ = model(template, search)
            torch.cuda.synchronize()

            print("[INFO] Running timed pass...")
            start_time = time.time()
            
            # 推理模式调用
            output = model(template, search)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
        print(f"[INFO] Inference successful.")
        print(f"[TIME] Latency: {(end_time - start_time)*1000:.2f} ms")
        
        # 5. Check Output
        # 我们最新的实现返回的是 (logits, boxes)
        if isinstance(output, tuple):
            pred_logits, pred_boxes = output
            print(f"[OUTPUT] Type: Tuple (Logits, Boxes)")
            print(f"    - Logits Shape: {pred_logits.shape}  (Expected: [B, 4, Vocab])")
            print(f"    - Boxes Shape:  {pred_boxes.shape}   (Expected: [B, 4, 4])")
            
            if pred_logits.shape[-1] == cfg.MODEL.HEAD.VOCAB_SIZE:
                print("[RESULT] ✅ TEST PASSED: Vocabulary size matches.")
            else:
                print(f"[RESULT] ❌ TEST FAILED: Expected vocab {cfg.MODEL.HEAD.VOCAB_SIZE}, got {pred_logits.shape[-1]}")
                
        elif isinstance(output, dict):
            # 训练模式下可能返回 dict
            print(f"[OUTPUT] Type: Dict (Keys: {output.keys()})")
        else:
            print(f"[OUTPUT] Type: Tensor (Shape: {output.shape})")

    except RuntimeError as e:
        print(f"[ERROR] Runtime Error during forward pass: {e}")
        if "mamba_ssm" in str(e):
            print(">>> Tip: This error is likely due to Mamba CUDA kernels compatibility.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()