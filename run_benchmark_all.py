import subprocess
import os
import json
import matplotlib.pyplot as plt
import numpy as np

def run_script(python_path, script_path):
    print(f"\n>>> Running {script_path} with {python_path}...")
    try:
        subprocess.run([python_path, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

def main():
    output_dir = './benchmark_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define environments and scripts
    # 1. ARTrackMamba (current env 'mtrack' or default python)
    python_mtrack = "python" # Assuming current env is mtrack
    script_mamba = "benchmark_models.py"
    
    # 2. ARTrackV2 (env 'artrack')
    python_artrack = "/root/miniconda3/envs/artrack/bin/python"
    script_v2 = "benchmark_artrack_v2.py"
    
    # Run benchmarks
    run_script(python_mtrack, script_mamba)
    run_script(python_artrack, script_v2)
    
    # Collect results
    results = {}
    
    # Load Mamba results
    mamba_json = os.path.join(output_dir, 'result_mamba.json')
    if os.path.exists(mamba_json):
        with open(mamba_json, 'r') as f:
            data = json.load(f)
            results['ARTrackMamba'] = data
            
    # Load V2 results
    v2_json = os.path.join(output_dir, 'result_v2.json')
    if os.path.exists(v2_json):
        with open(v2_json, 'r') as f:
            data = json.load(f)
            results['ARTrackV2'] = data
            
    if not results:
        print("No results found!")
        return

    # Unified Report
    print("\n" + "="*80)
    print(f"{'UNIFIED BENCHMARK REPORT':^80}")
    print("="*80)
    print(f"{'Model':<20} | {'Params (M)':<12} | {'FPS (GPU)':<12} | {'Lat (ms)':<12} | {'Mem (MB)':<12}")
    print("-" * 80)
    
    models = list(results.keys())
    metrics_gpu = {'FPS': [], 'Latency': [], 'Memory': []}
    params = []
    
    for model in models:
        data = results[model]
        res_gpu = data['results'].get('cuda', {})
        
        p = data.get('params_M', 0)
        fps = res_gpu.get('FPS', 0)
        lat = res_gpu.get('Latency', 0)
        mem = res_gpu.get('Memory', 0)
        
        print(f"{model:<20} | {p:<12.2f} | {fps:<12.2f} | {lat:<12.2f} | {mem:<12.2f}")
        
        params.append(p)
        metrics_gpu['FPS'].append(fps)
        metrics_gpu['Latency'].append(lat)
        metrics_gpu['Memory'].append(mem)
        
    print("-" * 80)
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Params
    axes[0].bar(models, params, color=['blue', 'orange'])
    axes[0].set_title('Parameters (M) - Lower is usually lighter')
    axes[0].set_ylabel('Million')
    
    # FPS
    axes[1].bar(models, metrics_gpu['FPS'], color=['blue', 'orange'])
    axes[1].set_title('Throughput (FPS) - Higher is Better')
    axes[1].set_ylabel('FPS')
    
    # Latency
    axes[2].bar(models, metrics_gpu['Latency'], color=['blue', 'orange'])
    axes[2].set_title('Latency (ms) - Lower is Better')
    axes[2].set_ylabel('ms')
    
    # Memory
    axes[3].bar(models, metrics_gpu['Memory'], color=['blue', 'orange'])
    axes[3].set_title('Peak GPU Memory (MB) - Lower is Better')
    axes[3].set_ylabel('MB')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'final_comparison.png')
    plt.savefig(save_path)
    print(f"\nFinal unified plot saved to {save_path}")

if __name__ == '__main__':
    main()
