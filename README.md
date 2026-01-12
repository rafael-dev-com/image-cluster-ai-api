# Image Cluster AI – API

AI-powered backend service for image clustering and semantic description, exposed through a REST API.

> **Proof of Concept (PoC)**  
> This project demonstrates a modular and extensible architecture for image analysis using AI.  
> It is not production-ready and is intentionally simplified to focus on design, clarity, and experimentation.

---

## 1. What is this API?

This API provides an **AI-driven pipeline** that:

1. Accepts multiple images
2. Extracts semantic embeddings using a deep learning model
3. Groups images using clustering algorithms
4. Generates a textual description for each cluster using a vision-language model

The result is a **semantic grouping of images**, not based on filenames or metadata, but on **visual meaning**.

---

## 2. Scope and goals

**In scope**
- Image ingestion via HTTP
- In-memory image processing
- AI-based embedding extraction
- Unsupervised clustering
- Automatic cluster description
- Clean separation of concerns (ports & adapters)

**Out of scope (by design)**
- Authentication / authorization
- Persistence (DB, object storage)
- Async task queues
- Horizontal scalability
- Production hardening

---

## 3. High-level architecture

This API follows a **Hexagonal / Ports & Adapters architecture**, allowing AI components to be replaced or extended without affecting the rest of the system.

```text
HTTP (FastAPI)
     │
     ▼
Application Service
     │
     ▼
Domain Pipeline
     │
     ├── Embedding Extraction (AI)
     ├── Clustering (ML)
     └── Caption Generation (AI)
     │
     ▼
Response Json

```
This diagram illustrates the end-to-end request flow, from the HTTP API layer to the domain pipeline, highlighting the AI and ML components involved in the image processing workflow.

---
## 4. Project structure

```text
image-cluster-ai-api/
├── src/
│   └── app/
│       ├── adapters/
│       │   └── web/            # FastAPI controllers
│       ├── application/        # Application services (use cases)
│       ├── domain/
│       │   ├── models/         # Core domain models
│       │   └── services/       # Pure domain logic
│       ├── ports/              # Interfaces (contracts)
│       ├── infrastructure/     # AI/ML implementations
│       ├── config/             # Runtime configuration
│       └── main.py
├── Dockerfile
├── requirements.txt
└── README.md
```
---

## 5. Core pipeline
The main pipeline executed by the API can be summarized as:

```text
[List[ImageItem]]
        │
        ▼
extract_embeddings
        │
        ▼
[List[EmbeddingVector]]
        │
        ▼
cluster_embeddings
        │
        ▼
[List[Cluster]]
        │
        ▼
generate_descriptions
        │
        ▼
[List[Cluster]]
```

Each step is isolated behind an interface, enabling easy replacement of implementations.

---

## 6. AI Components

### 6.1 Embedding Extraction (AI)

- Uses a deep learning vision model to convert images into vector representations
- Output: dense numerical embeddings
- Purpose: capture semantic similarity between images

This is the foundation that enables meaningful clustering.

### 6.2 Clustering (ML)

- Uses unsupervised clustering (e.g. cosine distance–based clustering)

- No predefined number of clusters

- Groups images based purely on embedding similarity

---

### 6.3 Caption generation (AI)

- Uses a vision-language model to generate a short textual description per cluster

- The description summarizes the common visual theme of the group

---

## 7. API Endpoints (simplified)

| Endpoint      | Method | Description |
|---------------|--------|-------------|
| `/cluster-images`    | POST   | Accepts multiple images and returns clustered results |
| `/health`     | GET    | Simple health check to verify that the API is running |

#### Request

| Endpoint     | Content-Type       | Body |
|-------------|------------------|------|
| `/cluster-images`  | `multipart/form-data` | Multiple image files |
| `/health`   | N/A               | N/A  |

#### Response

| Endpoint     | Field         | Type        | Description |
|-------------|---------------|-------------|-------------|
| `/cluster-images`  | `clusters`    | json string | Images grouped by cluster |
| `/health`   | `status`      | String      | `"ok"` if API is running |

Example response (`/cluster-images`):

```json
{
  "clusters": [
    {
      "name": 0,
      "image_ids": ["img1.jpg", "img2.jpg"],
      "description": "black leather shoes"
    }
  ]
}
```
---

## 9. Running the API locally (Full Deploy with Docker Compose)

You can run the **full deploy** (API + web frontend + test images) using **Docker Compose**.  

#### Recommended: Using Git (simplest)

1. Clone the deploy project:

```bash
git clone https://github.com/rafael-dev-com/image-cluster-ai-deploy
cd image-cluster-ai-deploy
```
---

2. Inside the deploy folder, clone the sub-repositories:
```bash
git clone https://github.com/rafael-dev-com/image-cluster-ai-api
git clone https://github.com/rafael-dev-com/image-cluster-ai-web
```
3. Start the services with Docker Compose:
```bash
docker-compose up --build
```
- This will build and start all services.

- The web client will run at http://localhost:8082.

- A folder with test images is included in the deploy structure for easy testing.

#### To stop all services:
```bash
docker-compose down
```
---

### Alternative: Manual download

If you prefer not to use git, you can also download the repositories manually:

- Place repo-api and repo-web inside the deploy folder.

- Make sure the folder structure matches:
```bash
image-cluster-ai-deploy/
├── image-cluster-ai-api/
├── image-cluster-ai-web/
└── docker-compose.yml
```
- Then, from inside the `image-cluster-ai-deploy` folder, run:
:
```bash
docker-compose up --build
```
The web client will run at http://localhost:8082.
This method works the same way; git just makes it more convenient.
---

## 10. Run only the API (Docker)

If you want to deploy only the API, follow these steps:

1. Clone the image-cluster-ai-api project:

```bash
git clone https://github.com/rafael-dev-com/image-cluster-ai-api
cd image-cluster-ai-api
```

1. Build the API Docker image:
```bash
docker build -t image-cluster-ai-api .
```
2. Run the API container:
```bash
docker run -p 8001:8001 --name image-cluster-ai-api image-cluster-ai-api
```
The API will be available at: http://localhost:8001

## 11. Related repositories

This API is part of a larger system:

- [Web client - image-cluster-ai-web](https://github.com/rafael-dev-com/image-cluster-ai-web)

- [Full deployment (API + Web + samples) - image-cluster-ai-deploy](https://github.com/rafael-dev-com/image-cluster-ai-deploy):

## 12. Future improvements

- Async processing for large batches

- Persistent storage

- Better error handling and observability

- Model versioning

- Configurable clustering strategies

- Production-ready security