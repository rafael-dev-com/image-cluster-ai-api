from abc import ABC, abstractmethod
from typing import List
from app.domain.models import Cluster

class CaptioningPort(ABC):
    """
    Abstract interface for generating descriptions for clusters.
    Implementations should update the `description` field of each Cluster in memory.
    """

    @abstractmethod
    def generate_descriptions(self, clusters: List[Cluster]) -> List[Cluster]:
        """
        Receives a list of Cluster objects and generates descriptions in memory.
        Updates the `description` field of each cluster.
        """
        pass
