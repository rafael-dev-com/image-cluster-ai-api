import numpy as np
import logging
from typing import List
from sklearn.metrics.pairwise import cosine_distances
import hdbscan

from app.domain.models import EmbeddingVector, Cluster
from app.ports.clustering_port import ClusteringPort

logger = logging.getLogger(__name__)

class HDBSCANClusteringAdapter(ClusteringPort):
    """
    Clustering adapter using HDBSCAN on embedding vectors.
    Groups ImageItems into Cluster objects based on cosine similarity.
    """

    def __init__(self, min_cluster_size=2, min_samples=1):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster_embeddings(self, embeddings: List[EmbeddingVector]) -> List[Cluster]:
        """
        Cluster embeddings and return a list of Cluster objects.
        Noise embeddings (label -1) are ignored.
        """
        if not embeddings:
            return []

        # Convert embeddings to a 2D numpy array
        embeddings_array = np.array([e.value for e in embeddings], dtype=np.float64)

        # Compute cosine distance matrix
        logger.info("Computing cosine distance matrix...")
        distance_matrix = cosine_distances(embeddings_array)

        # Apply HDBSCAN clustering
        logger.info("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed"
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Count clusters (excluding noise)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found clusters (excluding noise): {num_clusters}")

        # Group ImageItems by cluster label
        clusters_dict = {}
        for emb, label in zip(embeddings, labels):
            if label == -1:
                continue  # Skip noise
            clusters_dict.setdefault(label, []).append(emb.image)

        # Convert grouped items into Cluster objects
        clusters: List[Cluster] = [
            Cluster(label=label, images=imgs, description=None)
            for label, imgs in clusters_dict.items()
        ]

        return clusters
