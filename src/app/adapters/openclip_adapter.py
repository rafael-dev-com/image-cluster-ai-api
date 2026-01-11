from typing import List
import torch
import open_clip
from tqdm import tqdm
import logging

from app.config.settings import DEVICE
from app.domain.models import ImageItem, EmbeddingVector
from app.ports.embedding_port import EmbeddingPort

logger = logging.getLogger(__name__)

class OpenCLIPEmbeddingAdapter(EmbeddingPort):
    """
    Embedding service using OpenCLIP.
    Supports GPU/MPS batching and CPU sequential processing.
    Logs progress during embedding extraction.
    """

    def __init__(self, batch_size_gpu: int = 16):
        # Initialize OpenCLIP model and preprocessing
        logger.info(f"Loading OpenCLIP model on {DEVICE}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=DEVICE
        )
        self.model.eval()
        self.batch_size_gpu = batch_size_gpu
        logger.info("OpenCLIP model loaded successfully.")

    def extract_embeddings(self, images: List[ImageItem]) -> List[EmbeddingVector]:
        """
        Extract embeddings for a list of ImageItem objects.
        Returns a list of EmbeddingVector objects with normalized vectors.
        """
        embeddings: List[EmbeddingVector] = []
        logger.info(f"Starting extraction of embeddings for {len(images)} images using {DEVICE}.")

        if DEVICE == "cpu":
            logger.info("Processing images sequentially on CPU...")
            for img in tqdm(images, desc="Extracting embeddings (CPU)"):
                tensor = self.preprocess(img.data).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = self.model.encode_image(tensor)
                    emb /= emb.norm(dim=-1, keepdim=True)  # Normalize vector
                embeddings.append(EmbeddingVector(image=img, value=emb.cpu().numpy()[0]))
        else:
            logger.info(f"Processing images in batches of {self.batch_size_gpu} on {DEVICE}...")
            for i in tqdm(range(0, len(images), self.batch_size_gpu), desc="Extracting embeddings (GPU)"):
                batch = images[i:i + self.batch_size_gpu]
                batch_tensor = torch.stack([self.preprocess(img.data) for img in batch]).to(DEVICE)

                with torch.no_grad():
                    emb = self.model.encode_image(batch_tensor)
                    emb /= emb.norm(dim=-1, keepdim=True)  # Normalize each vector

                # Convert each embedding to EmbeddingVector with its ImageItem
                for img_item, e in zip(batch, emb.cpu().numpy()):
                    embeddings.append(EmbeddingVector(image=img_item, value=e))

        logger.info(f"Completed extraction of {len(embeddings)} embeddings.")
        return embeddings
