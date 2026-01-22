import os
from typing import List
from app.domain.models import Cluster
from app.ports.storage_port import StoragePort

class DiskStorageAdapter(StoragePort):
    """
    Storage adapter that saves clusters and their images/descriptions to disk.
    Each cluster is stored in a separate folder.
    """

    def __init__(self, output_folder: str):
        """
        output_folder: Root folder where clusters will be saved.
        """
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def save(self, clusters: List[Cluster]) -> None:
        """
        Save images and descriptions of clusters to disk.
        Each cluster is saved in a subfolder "cluster_<label>".
        Noise cluster (label=-1) is saved in "cluster_noise".
        """
        for cluster in clusters:
            folder_name = f"cluster_{cluster.label}" if cluster.label != -1 else "cluster_noise"
            path = os.path.join(self.output_folder, folder_name)
            os.makedirs(path, exist_ok=True)

            # Save images using their original ID as filename
            for img_obj in cluster.images:
                img_path = os.path.join(path, img_obj.id)
                img_obj.data.save(img_path)

            # Save description if it exists
            if cluster.description:
                desc_path = os.path.join(path, "description.txt")
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write(cluster.description)
