from dataclasses import dataclass
from typing import List
import numpy as np
from PIL import Image

# -------------------------------
# Image item in memory
# -------------------------------
@dataclass
class ImageItem:
    """Represents an image in memory with an ID."""
    id: str
    data: Image.Image

# -------------------------------
# Embeddings
# -------------------------------
@dataclass
class EmbeddingVector:
    """Represents the embedding of a single image."""
    image: ImageItem
    value: np.ndarray

# -------------------------------
# Clusters
# -------------------------------
@dataclass
class Cluster:
    """Represents a cluster of images."""
    label: int
    images: List[ImageItem]
    description: str = None
