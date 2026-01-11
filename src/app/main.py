import os
from typing import List
from PIL import Image
import logging
from app.adapters.blip_adapter import BLIPCaptioningAdapter
from app.adapters.disk_storage_adapter import DiskStorageAdapter
from app.adapters.hdbscan_adapter import HDBSCANClusteringAdapter
from app.adapters.json_renderer_adapter import JsonRendererAdapter
from app.adapters.openclip_adapter import OpenCLIPEmbeddingAdapter
from app.config.settings import OUTPUT_FOLDER, IMAGE_FOLDER
from app.core.rchestrator import run_pipeline
from app.domain.models import ImageItem, Cluster

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Functions / "Microservices" ---
def get_images(image_folder=IMAGE_FOLDER) -> list[ImageItem]:
    image_paths = load_images(image_folder)
    images_in_memory = [Image.open(p).convert("RGB") for p in image_paths]
    images_id = [os.path.basename(p) for p in image_paths]  # file names only

    images_objs: list[ImageItem] = [
        ImageItem(id=name, data=img)
        for name, img in zip(images_id, images_in_memory)
    ]
    return images_objs


def load_images(folder_path):
    """Return list of valid image paths."""
    logger.info(f"Loading images from '{folder_path}'...")
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_paths:
        raise FileNotFoundError("No images found.")
    return image_paths


# --- Orchestrator ---
def main():
    storage = DiskStorageAdapter(OUTPUT_FOLDER)
    renderer = JsonRendererAdapter()

    clusters: List[Cluster] = run_pipeline(get_images(), OpenCLIPEmbeddingAdapter(), HDBSCANClusteringAdapter(),
                                           BLIPCaptioningAdapter())

    storage.save(clusters)

    # Generate JSON string
    json_str = renderer.render(clusters)

    logger.info(f"Clustering and description completed. Results in '{OUTPUT_FOLDER}'")


if __name__ == "__main__":
    main()
