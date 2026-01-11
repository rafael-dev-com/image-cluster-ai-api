import json
from typing import List

from app.domain.models import Cluster
from app.ports.render_port import RendererPort

class JsonRendererAdapter(RendererPort):
    """
    Renderer adapter that converts Cluster objects into a JSON string.
    """

    def render(self, clusters: List[Cluster]) -> str:
        # Prepare a list of cluster dictionaries
        clusters_list = []

        for cluster in clusters:
            cluster_entry = {
                "name": str(cluster.label),
                "image_ids": [img.id for img in cluster.images],  # Collect image IDs
                "description": cluster.description
            }
            clusters_list.append(cluster_entry)

        # Convert list of clusters to a JSON string
        json_string = json.dumps(
            {"clusters": clusters_list},
            indent=4,
            ensure_ascii=False
        )

        return json_string
