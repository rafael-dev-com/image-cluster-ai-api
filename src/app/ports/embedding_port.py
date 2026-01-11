from abc import ABC, abstractmethod
from typing import List
from app.domain.models import ImageItem, EmbeddingVector

class EmbeddingPort(ABC):
    """
    Abstract interface for extracting embeddings from images.
    Implementations should return a list of EmbeddingVector objects
    corresponding to the input ImageItem objects.
    """

    @abstractmethod
    def extract_embeddings(self, images: List[ImageItem]) -> List[EmbeddingVector]:
        """
        Extract embeddings for a list of images.
        Each EmbeddingVector contains the associated ImageItem and its embedding vector.
        """
        pass
