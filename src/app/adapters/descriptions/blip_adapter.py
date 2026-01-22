import torch
from typing import List
import logging

from transformers import BlipProcessor, BlipForConditionalGeneration

from app.domain.models import Cluster
from app.ports.captioning_port import CaptioningPort
from app.config.settings import DEVICE

logger = logging.getLogger(__name__)

class BLIPCaptioningAdapter(CaptioningPort):
    """
    Captioning adapter using BLIP to generate descriptions for clusters.
    Updates the `description` field of each Cluster in memory.
    """
    def __init__(self):
        """
        Load BLIP model and processor into memory.
        """
        logger.info("Loading BLIP model and processor...")
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            use_fast=True
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(DEVICE)
        logger.info("BLIP loaded successfully.")

    def generate_descriptions(self, clusters: List[Cluster]) -> List[Cluster]:
        # Generate descriptions for each cluster
        logger.info("Generating descriptions with BLIP...")

        for cluster in clusters:
            imgs = cluster.images
            # Skip empty clusters or unlabelled (-1) clusters
            if cluster.label != -1 and imgs:
                # Take up to first 3 images for description
                selected_imgs = imgs[:3]
                descriptions = []

                for img_obj in selected_imgs:
                    # Ensure image is RGB
                    img = img_obj.data.convert("RGB")
                    # Prepare inputs for BLIP
                    inputs = self.processor(
                        images=img, return_tensors="pt"
                    ).to(DEVICE)

                    # Generate description without computing gradients
                    with torch.no_grad():
                        out = self.model.generate(**inputs)

                    # Decode output tokens to text
                    desc = self.processor.decode(out[0], skip_special_tokens=True)
                    descriptions.append(desc)

                # Remove duplicates and join descriptions
                cluster.description = " / ".join(dict.fromkeys(descriptions))
                logger.info(f"Cluster {cluster.label} description: {cluster.description}")

        return clusters
