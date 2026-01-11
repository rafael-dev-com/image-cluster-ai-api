from abc import ABC, abstractmethod
from typing import List
from app.domain.models import Cluster

class StoragePort(ABC):
    """
    Abstract interface for storing clusters.
    Implementations should handle persistence of Cluster objects.
    """

    @abstractmethod
    def save(self, clusters: List[Cluster]):
        """
        Save a list of Cluster objects to storage.
        """
        pass
