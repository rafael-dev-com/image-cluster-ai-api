import json
from io import BytesIO
from typing import List

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.adapters.blip_adapter import BLIPCaptioningAdapter
from app.adapters.disk_storage_adapter import DiskStorageAdapter
from app.adapters.hdbscan_adapter import HDBSCANClusteringAdapter
from app.adapters.json_renderer_adapter import JsonRendererAdapter
from app.adapters.openclip_adapter import OpenCLIPEmbeddingAdapter
from app.config.settings import OUTPUT_FOLDER
from app.core.rchestrator import run_pipeline
from app.domain.models import ImageItem, Cluster

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum number of images allowed
MAX_IMAGES = 24

# Max file size in bytes (e.g., 2 MB)
MAX_FILE_SIZE = 2 * 1024 * 1024

@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Simple health check to verify that the API is running.
    Returns status OK in JSON format.
    """
    return JSONResponse(content={"status": "ok"})

@app.post("/cluster-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Receives a list of images, clusters them, and returns JSON results.
    """

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if len(files) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files uploaded. Maximum allowed is {MAX_IMAGES}."
        )

    images: List[ImageItem] = []

    for file in files:
        # Check file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {file.filename}")

        # Read file contents
        contents = await file.read()

        # Check file size
        if len(contents) > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File too large: {file.filename}. Maximum allowed size is {max_mb:.1f} MB.")

        try:
            # Convert to PIL.Image
            img = Image.open(BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail=f"Cannot open image: {file.filename}")

        # Create ImageItem
        images.append(ImageItem(id=file.filename, data=img))

    # Initialize adapters
    storage = DiskStorageAdapter(OUTPUT_FOLDER)
    renderer = JsonRendererAdapter()

    # Run pipeline
    clusters: List[Cluster] = run_pipeline(
        images, OpenCLIPEmbeddingAdapter(), HDBSCANClusteringAdapter(), BLIPCaptioningAdapter()
    )

    # Save results to disk
    # storage.save(clusters)

    # Generate JSON string
    json_str = renderer.render(clusters)

    return JSONResponse(content=json.loads(json_str))
