from abc import ABC
from typing import List

from app.domain.models import EmbeddingVector, Cluster

class ClusteringPort(ABC):
    """
    Abstract interface for clustering embeddings.
    Implementations should group ImageItems into Cluster objects based on similarity.
    """

    def cluster_embeddings(self, embeddings: List[EmbeddingVector]) -> List[Cluster]:
        """
        Receive a list of embeddings and return a list of Cluster objects
        with ImageItems grouped according to similarity.
        """
        pass
