from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import pandas as pd

router = APIRouter(prefix="/vision", tags=["Vision"])


@router.post("/edge-detection")
async def edge_detection(file: UploadFile = File(...)):

    image = await file.read()

    image = np.frombuffer(image, np.uint8)

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    edge_pixels = int(np.count_nonzero(edges))

    return {
        "message": "Edge detection completed",
        "edge_pixels": edge_pixels
    }