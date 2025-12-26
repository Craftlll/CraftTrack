import torch
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    print("✅ Mamba CUDA kernel loaded successfully.")
except ImportError:
    print("❌ Mamba CUDA kernel NOT found. Using slow fallback!")
except Exception as e:
    print(f"❌ Error loading Mamba: {e}")

# 检查 CUDA 是否可用
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")