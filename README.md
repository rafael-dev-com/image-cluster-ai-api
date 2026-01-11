# Image Clustering and Captioning Pipeline

> **Note:** This project is a **proof of concept**. The architecture is modular and works, but there is room for improvements, optimizations, and handling more complex cases.  
> This project leverages **Artificial Intelligence (AI)** for processing images, extracting embeddings, clustering them, and generating textual descriptions automatically.

This Python project allows you to **cluster similar images** using **OpenCLIP embeddings**, generate automatic descriptions for each cluster with **BLIP**, and export results as **JSON**. It is designed to integrate easily with frontends via **FastAPI**.

---

## Features

- Use **AI-powered embeddings** with OpenCLIP (ViT-B-32) to represent images numerically.  
- Cluster similar images using **HDBSCAN** with cosine distance.  
- Generate **AI-generated descriptions** for each cluster using BLIP.  
- Return results as **JSON** ready for frontend consumption.  
- **Modular architecture** based on ports and adapters for easy model replacement.  
- Designed as a **proof of concept** for AI-based image clustering and captioning.

---

## Requirements

- Python 3.11+  
- GPU recommended for faster AI processing (CUDA or MPS).  
- Install dependencies via `requirements.txt`.

---

