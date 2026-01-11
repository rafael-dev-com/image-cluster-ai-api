from abc import ABC, abstractmethod
from typing import List
from app.domain.models import Cluster

class RendererPort(ABC):
    """
    Abstract interface for rendering clusters.
    Implementations should convert Cluster objects into a string format (e.g., JSON).
    """

    @abstractmethod
    def render(self, clusters: List[Cluster]) -> str:
        """
        Render a list of Cluster objects into a string.
        """
        pass
