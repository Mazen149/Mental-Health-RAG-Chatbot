import os
from pathlib import Path
from huggingface_hub import snapshot_download
from ..config import config

def download_artifacts() -> None:
    """
    Downloads the artifact directory from Hugging Face if the necessary model files are missing.
    """
    repo_id = config.HF_ARTIFACTS_REPO
    if not repo_id:
        print("--> [Downloader] Skipping download. HF_ARTIFACTS_REPO is not set in environment.")
        return

    # Check if a critical file exists to determine if download is needed.
    # We check for the ONNX model as the primary indicator.
    model_path = config.MOD2_DIR / "model.onnx"
    
    if model_path.exists():
        print(f"--> [Downloader] Artifacts already exist locally at {config.ARTIFACTS_DIR}.")
        return

    print(f"--> [Downloader] Missing artifacts locally. Downloading from Hugging Face: {repo_id}...")
    
    # We will download the repository contents directly into the artifacts directory
    # huggingface_hub will use its cache and symlink if possible, or we can use local_dir.
    # using local_dir ensures the files are exactly where the app expects them.
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(config.ARTIFACTS_DIR),
            local_dir_use_symlinks=False,
            token=config.HF_TOKEN
        )
        print("--> [Downloader] Successfully downloaded artifacts.")
    except Exception as e:
        print(f"--> [Downloader] ERROR: Failed to download artifacts from {repo_id}. Exception: {e}")
        import sys
        sys.exit(1)
