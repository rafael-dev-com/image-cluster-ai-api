import torch
from typing import List
from tqdm import tqdm
import logging

from app.config.settings import DEVICE
from app.domain.models import ImageItem, EmbeddingVector
from app.ports.embedding_port import EmbeddingPort

logger = logging.getLogger(__name__)

class DINOv2EmbeddingAdapter(EmbeddingPort):
    """
    Embedding service using DINOv2.
    Extracts normalized embeddings for a list of ImageItem objects.
    Supports GPU/MPS batching and CPU sequential processing.
    """

    def __init__(self, batch_size_gpu: int = 16):
        logger.info(f"Loading DINOv2 model on {DEVICE}...")
        # Carga del modelo DINOv2 ViT-B/14
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitb14"
        ).to(DEVICE)
        self.model.eval()
        self.batch_size_gpu = batch_size_gpu
        logger.info("DINOv2 model loaded successfully.")

        # Preprocess: se usa transform de DINOv2
        # Normalmente DINOv2 espera imágenes [0,1] y tamaño 224x224
        import torchvision.transforms as T
        self.preprocess = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def extract_embeddings(self, images: List[ImageItem]) -> List[EmbeddingVector]:
        embeddings: List[EmbeddingVector] = []
        logger.info(f"Starting extraction of embeddings for {len(images)} images using {DEVICE}.")

        # Procesamiento por batch
        for i in tqdm(range(0, len(images), self.batch_size_gpu), desc="Extracting embeddings"):
            batch = images[i:i + self.batch_size_gpu]

            # Preprocess y apilado en tensor
            batch_tensor = torch.stack([self.preprocess(img.data) for img in batch]).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                emb = self.model(batch_tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalización

            # Convertir a EmbeddingVector
            for img_item, e in zip(batch, emb.cpu().numpy()):
                embeddings.append(EmbeddingVector(image=img_item, value=e))

        logger.info(f"Completed extraction of {len(embeddings)} embeddings.")
        return embeddings
