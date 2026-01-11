from datetime import datetime

import torch

# --- Device ---
def get_device() -> str:
    """
    Select the best available device:
    - CUDA: NVIDIA GPUs
    - MPS: Apple Silicon
    - CPU: fallback
    """
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device()
IMAGE_FOLDER = "../../input"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FOLDER = f"../../../output_{timestamp}"