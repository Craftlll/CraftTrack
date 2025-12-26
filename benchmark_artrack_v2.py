import torch
import time
import os
import argparse
# from thop import profile
# from thop.utils import clever_format

# Add current directory to path
import sys
sys.path.append(os.getcwd())

# Import model builders
from lib.models.artrackv2_seq.artrackv2_seq import build_artrackv2_seq
from lib.config.artrackv2_seq.config import cfg, update_config_from_file

def measure_throughput(model, dummy_inputs, device, n_warmup=10, n_repeat=50):
    model.eval()
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(**dummy_inputs)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(n_repeat):
        with torch.no_grad():
            _ = model(**dummy_inputs)
            
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / n_repeat
    fps = 1.0 / avg_time
    return fps, avg_time * 1000 # ms

def measure_memory(model, dummy_inputs, device):
    if device != 'cuda':
        return 0.0
        
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        _ = model(**dummy_inputs)
        
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    return max_mem

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dummy_inputs(cfg, device='cuda'):
    B = 1
    T_H, T_W = cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.SIZE
    S_H, S_W = cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.SIZE
    
    # ARTrackV2 expects template as [B, T, C, H, W] usually (where T=1 or 2)
    # The code slices template[:, 0], so we need at least 1 frame dim.
    template = torch.randn(B, 1, 3, T_H, T_W).to(device)
    search = torch.randn(B, 3, S_H, S_W).to(device)
    
    # Check embed dim
    embed_dim = 768 # Default for Base
    if hasattr(cfg.MODEL.BACKBONE, 'EMBED_DIM'):
        embed_dim = cfg.MODEL.BACKBONE.EMBED_DIM
    elif 'base' in cfg.MODEL.BACKBONE.TYPE:
        embed_dim = 768
        
    # Dummy dynamic template feature [B, 64, C] (8x8 patch grid for 128x128 image with patch 16)
    dz_feat = torch.randn(B, 64, embed_dim).to(device)
    
    # Dummy sequence input [B, 0]
    seq_input = torch.zeros(B, 0).to(device)
    
    return {
        'template': template,
        'search': search,
        'dz_feat': dz_feat,
        'seq_input': seq_input
    }

def main():
    print("Benchmarking ARTrackV2 in 'artrack' environment...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Setup Config & Model
    config_file = "experiments/artrackv2_seq/artrackv2_seq_256_full.yaml"
    if os.path.exists(config_file):
        update_config_from_file(config_file)
        print(f"Loaded config from {config_file}")
    else:
        print(f"Config file {config_file} not found, using default config.")
        # Ensure minimal defaults
        cfg.MODEL.BACKBONE.TYPE = 'vit_base_patch16_224'
    
    # Build Model
    # Note: Training=False to avoid loading optimizer parts
    try:
        model = build_artrackv2_seq(cfg, training=False)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error building model: {e}")
        return
    
    # 2. Parameters
    n_params = count_parameters(model)
    print(f"\nModel Parameters: {n_params / 1e6:.2f} M")
    
    # 3. Inputs
    inputs = get_dummy_inputs(cfg, device)
    
    # 4. Speed & Memory
    print(f"\n--- Speed Benchmark ({device}) ---")
    results = {}
    try:
        fps, latency = measure_throughput(model, inputs, device)
        print(f"FPS: {fps:.2f}")
        print(f"Latency: {latency:.2f} ms")
        results['FPS'] = fps
        results['Latency'] = latency
        
        if device == 'cuda':
            mem = measure_memory(model, inputs, device)
            print(f"Peak Memory: {mem:.2f} MB")
            results['Memory'] = mem
    except Exception as e:
        print(f"Error during benchmark: {e}")
        
    # Save JSON for master script
    import json
    output_dir = './benchmark_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_data = {
        'model': 'ARTrackV2',
        'params_M': n_params / 1e6,
        'results': {device: results}
    }
    
    with open(os.path.join(output_dir, 'result_v2.json'), 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"\nJSON result saved to {os.path.join(output_dir, 'result_v2.json')}")

if __name__ == '__main__':
    main()
