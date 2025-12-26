import torch
import time
import os
import argparse
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import numpy as np

# Import model builders
from lib.models.artrackmamba_seq.artrackmamba_seq import build_artrackmamba_seq
from lib.models.artrackv2_seq.artrackv2_seq import build_artrackv2_seq

# Import config helper
from lib.config.artrackmamba_seq.config import update_config_from_file as update_config_mamba
# from lib.config.artrackv2_seq.config import update_config_from_file as update_config_v2

# Import configs
from lib.config.artrackmamba_seq.config import cfg as cfg_mamba
from lib.config.artrackv2_seq.config import cfg as cfg_v2

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
        return 0.0 # Cannot easily measure CPU memory in same way
        
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        _ = model(**dummy_inputs)
        
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    return max_mem

def get_dummy_inputs(cfg, device='cuda', is_v2=False):
    # Basic inputs: template, search
    # Template: [B, 3, 128, 128] -> [B, 1, 3, 128, 128] for seq models usually
    # Search: [B, 3, 256, 256]
    B = 1
    T_H, T_W = cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.SIZE
    S_H, S_W = cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.SIZE
    
    template = torch.randn(B, 1, 3, T_H, T_W).to(device)
    search = torch.randn(B, 3, S_H, S_W).to(device)
    
    inputs = {
        'template': template,
        'search': search
    }
    
    if is_v2:
        # ARTrackV2 requires explicit None for some optional args or dummy seq_input
        # Based on error: seqs_input_ = torch.cat([trajectory, command], dim=1)
        # It seems it expects seq_input to be a dict or tensor, not None
        # Let's check signature or just provide dummy if needed.
        # ARTrackV2 forward: template, dz_feat, search, seq_input=None
        # The error implies seq_input cannot be None inside backbone if it tries to concat it.
        # Let's look at base_backbone.py logic later if needed.
        # For now, let's try providing minimal dummy seq_input
        # But wait, usually seq_input is optional. Let's check artrackv2_seq.py call:
        # self.backbone(..., seqs_input=seq_input)
        # If seq_input is None, it might fail.
        # Let's create a dummy command and trajectory
        # inputs['seq_input'] = {
        #     'trajectory': torch.zeros(B, 0, 4).to(device), # Empty or dummy
        #     'command': torch.zeros(B, 0, 4).to(device) # Empty or dummy
        # }
        # Actually, let's look at the error again: "expected Tensor as element 0... but got NoneType"
        # It means `trajectory` variable is None.
        # This likely happens when seqs_input is None.
        # The code at line 145 says: trajectory = seqs_input
        # So we must pass a Tensor for seqs_input
        inputs['seq_input'] = torch.zeros(B, 0).to(device) # Assuming empty sequence for now, but dim=1 concat might fail if 0
        # Wait, concat([trajectory, command], dim=1). Command is [B, 5]. Trajectory is [B, L].
        # If trajectory is [B, 0], it works.
        
        # New Error: 'NoneType' object has no attribute 'shape' at len_z = z_0.shape[1] + z_1.shape[1]
        # This means z_1 is None. z_1 comes from dz_feat.
        # ARTrackV2 expects dz_feat to be a tensor if it uses it.
        # If we pass None, we need to handle it or pass a dummy tensor.
        # z_1 is the dynamic template feature.
        # Usually it has shape [B, L_z, C].
        # Let's check z_0 shape. Template is [B, 3, 128, 128] -> patch embed -> [B, 64, C] (if patch 16).
        # So z_1 should probably be [B, 64, C] or similar.
        # However, BaseBackbone L154: z_0 = self.patch_embed(z_0).
        # And L155: z_1 = z_1_feat.
        # So z_1_feat must be already embedded?
        # Let's check ARTrackV2.forward:
        # z_1_feat=dz_feat.
        # So yes, dz_feat must be provided.
        # Let's create a dummy dz_feat.
        # It should match z_0 spatial dim (8x8=64).
        # C is embed_dim (usually 768 for Base).
        # Let's infer C from config or assume 768.
        # Or better, let the model process a dummy image through patch_embed if possible?
        # No, forward expects z_1_feat.
        # Let's check where z_1 comes from in real tracking.
        # It comes from memory/template update.
        # For benchmark, we can just use a random tensor.
        embed_dim = 768 # Vit-Base
        # Check cfg_v2.MODEL.BACKBONE.EMBED_DIM if available, else 768
        if hasattr(cfg_v2.MODEL.BACKBONE, 'EMBED_DIM'):
            embed_dim = cfg_v2.MODEL.BACKBONE.EMBED_DIM
        
        inputs['dz_feat'] = torch.randn(B, 64, embed_dim).to(device)
        
    return inputs

def main():
    parser = argparse.ArgumentParser(description='Benchmark ARTrackMamba vs ARTrackV2')
    parser.add_argument('--output_dir', default='./benchmark_results', help='Output directory for plots')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    devices = []
    if torch.cuda.is_available():
        devices.append('cuda')
    # devices.append('cpu') # Mamba SSM likely requires CUDA
    print("Warning: Skipping CPU benchmark as Mamba SSM usually requires CUDA kernels.")
    
    results = {} # {device: {metric: val}}
    
    for device in devices:
        print(f"\nBenchmarking on {device}...")
        results[device] = {}
        
        # 1. Setup Models
        print("Building ARTrackMamba...")
        # cfg_mamba.merge_from_file("experiments/artrackmamba_seq/artrackmamba_seq_256_base.yaml")
        update_config_mamba("experiments/artrackmamba_seq/artrackmamba_seq_256_base.yaml")
        # Force minimal config for inference if needed
        model_mamba = build_artrackmamba_seq(cfg_mamba)
        model_mamba.to(device)
        
        # 2. Prepare Inputs
        inputs_mamba = get_dummy_inputs(cfg_mamba, device)
        
        # 3. Run Benchmark
        print(f"--- Speed Benchmark (FPS) [{device}] ---")
        # Reduce repeat for CPU to save time
        n_repeat = 50 if device == 'cuda' else 10
        fps_mamba, lat_mamba = measure_throughput(model_mamba, inputs_mamba, device, n_repeat=n_repeat)
        results[device]['FPS'] = fps_mamba
        results[device]['Latency'] = lat_mamba
        print(f"ARTrackMamba ({device}): {fps_mamba:.2f} FPS ({lat_mamba:.2f} ms)")
        
        print(f"--- Memory Benchmark (Peak MB) [{device}] ---")
        mem_mamba = measure_memory(model_mamba, inputs_mamba, device)
        results[device]['Memory'] = mem_mamba
        if device == 'cuda':
            print(f"ARTrackMamba ({device}): {mem_mamba:.2f} MB")
        else:
            print(f"ARTrackMamba ({device}): N/A (RAM usage skipped)")

    # 4. Visualize
    metrics = ['FPS', 'Latency']
    
    # Plot FPS Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(devices))
    fps_values = [results[d]['FPS'] for d in devices]
    plt.bar(x, fps_values, color=['blue', 'green'])
    plt.xticks(x, devices)
    plt.title('Throughput (FPS) - Higher is Better')
    plt.ylabel('FPS')
    
    # Plot Latency Comparison
    plt.subplot(1, 2, 2)
    lat_values = [results[d]['Latency'] for d in devices]
    plt.bar(x, lat_values, color=['blue', 'green'])
    plt.xticks(x, devices)
    plt.title('Latency (ms) - Lower is Better')
    plt.ylabel('Time (ms)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'benchmark_device_comparison.png'))
    print(f"\nResults plot saved to {os.path.join(args.output_dir, 'benchmark_device_comparison.png')}")
    
    # Print Summary Table
    print("\n=== Benchmark Summary ===")
    print(f"{'Device':<10} | {'FPS':<15} | {'Latency (ms)':<15} | {'Memory (MB)':<15}")
    print("-" * 65)
    
    for device in devices:
        mem_str = f"{results[device]['Memory']:.2f}" if device == 'cuda' else "N/A"
        print(f"{device:<10} | {results[device]['FPS']:<15.2f} | {results[device]['Latency']:<15.2f} | {mem_str:<15}")
    print("-" * 65)

if __name__ == '__main__':
    main()
