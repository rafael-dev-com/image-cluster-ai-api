from typing import List
import logging

from app.domain.models import ImageItem, EmbeddingVector, Cluster
from app.ports.captioning_port import CaptioningPort
from app.ports.clustering_port import ClusteringPort
from app.ports.embedding_port import EmbeddingPort

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def run_pipeline(
        images: List[ImageItem],
        embedding_service: EmbeddingPort,
        clustering_service: ClusteringPort,
        captioning_service: CaptioningPort
) -> List[Cluster]:
    """
    Run the full pipeline: extract embeddings, cluster images, and generate descriptions.
    """
    logger.info(f"Starting pipeline with {len(images)} images.")

    # Step 1: Extract embeddings
    logger.info("Extracting embeddings...")
    embeddings: List[EmbeddingVector] = embedding_service.extract_embeddings(images)
    logger.info(f"Extracted {len(embeddings)} embeddings.")

    # Step 2: Cluster embeddings
    logger.info("Clustering embeddings...")
    clusters: List[Cluster] = clustering_service.cluster_embeddings(embeddings)
    logger.info(f"Generated {len(clusters)} clusters.")

    # Step 3: Generate descriptions
    logger.info("Generating descriptions for clusters...")
    clusters = captioning_service.generate_descriptions(clusters)
    logger.info("Pipeline completed successfully.")

    return clusters
