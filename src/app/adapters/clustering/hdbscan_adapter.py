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
    Ensures no image is left out, even if considered noise by HDBSCAN.
    """

    def __init__(self, min_cluster_size: int = 2, min_samples: int = 1):
        """
        min_cluster_size must be >= 2 (HDBSCAN requirement)
        Images marked as noise (-1) will get their own cluster automatically.
        """
        if min_cluster_size < 2:
            logger.warning(
                "HDBSCAN min_cluster_size must be >= 2, automatically setting to 2"
            )
            min_cluster_size = 2

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def cluster_embeddings(self, embeddings: List[EmbeddingVector]) -> List[Cluster]:
        if not embeddings:
            return []

        # Convert embeddings to 2D numpy array
        embeddings_array = np.array([e.value for e in embeddings], dtype=np.float64)

        # Compute cosine distance matrix
        logger.info("Computing cosine distance matrix...")
        distance_matrix = cosine_distances(embeddings_array)

        # Apply HDBSCAN clustering
        logger.info("Clustering with HDBSCAN (instance-level settings)...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="precomputed",
            cluster_selection_method="leaf",
            prediction_data=True
        )

        labels = clusterer.fit_predict(distance_matrix)

        # Prepare dictionary to group images by cluster label
        clusters_dict = {}
        next_label = max(labels) + 1 if len(labels) > 0 else 0

        for emb, label in zip(embeddings, labels):
            # Assign noise images (-1) to their own cluster
            if label == -1:
                label = next_label
                next_label += 1
            clusters_dict.setdefault(label, []).append(emb.image)

        # Convert to Cluster objects
        clusters: List[Cluster] = [
            Cluster(label=label, images=imgs, description=None)
            for label, imgs in clusters_dict.items()
        ]

        logger.info(f"Total clusters (including previously noise images): {len(clusters)}")
        return clusters