import os
from huggingface_hub import hf_hub_download
from tqdm import tqdm

def download_models():
    # Define the directory to save the models
    save_dir = "/root/CraftTrack/pretrained_models/vim"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting download of Vision Mamba (Vim) pretrained weights to {save_dir}...")
    
    models = [
        {
            "name": "Vim-tiny",
            "repo_id": "hustvl/Vim-tiny-midclstok",
            "filename": "vim_t_midclstok_76p1acc.pth"
        },
        {
            "name": "Vim-small",
            "repo_id": "hustvl/Vim-small-midclstok",
            "filename": "vim_s_midclstok_80p5acc.pth"
        },
        {
            "name": "Vim-base",
            "repo_id": "hustvl/Vim-base-midclstok",
            "filename": "vim_b_midclstok_81p9acc.pth"
        }
    ]
    
    for model in models:
        print(f"\nDownloading {model['name']}...")
        try:
            # Check if file already exists
            file_path = os.path.join(save_dir, model['filename'])
            if os.path.exists(file_path):
                print(f"{model['name']} already exists at {file_path}")
                continue

            # Download with progress bar using standard library if tqdm fails or internal hf implementation
            # hf_hub_download automatically shows a progress bar if not disabled.
            # But to be explicit and ensure visibility:
            file_path = hf_hub_download(
                repo_id=model['repo_id'],
                filename=model['filename'],
                local_dir=save_dir,
                resume_download=True
            )
            print(f"Successfully downloaded {model['name']} to: {file_path}")
        except Exception as e:
            print(f"Failed to download {model['name']}: {e}")

if __name__ == "__main__":
    # Ensure huggingface_hub is installed
    try:
        import huggingface_hub
        # Try to import tqdm, if not available, hf_hub_download still has its own default progress bar
        try:
            import tqdm
        except ImportError:
            print("tqdm not found, installing...")
            os.system("pip install tqdm")
            
        download_models()
    except ImportError:
        print("Please install huggingface_hub first: pip install huggingface_hub")
