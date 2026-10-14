from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.classifier import classify_image

router = APIRouter()


class ImageRequest(BaseModel):
    imageUrl: str


class TextRequest(BaseModel):
    description: str


@router.post("/classify-image")
async def classify_image_endpoint(req: ImageRequest):
    """
    Classify an image into a civic category using CLIP.
    """
    if not req.imageUrl:
        raise HTTPException(status_code=400, detail="imageUrl is required")

    result = classify_image(req.imageUrl)
    return result


@router.post("/analyze-text")
async def analyze_text_endpoint(req: TextRequest):
    """
    Simple keyword-based text analysis (placeholder for now).
    We'll add Hugging Face NLP here later.
    """
    text = (req.description or "").lower()

    category = "other"
    if any(k in text for k in ["pothole", "gaddha", "pit", "hole"]):
        category = "pothole"
    elif any(k in text for k in ["garbage", "trash", "waste", "kachra", "dump"]):
        category = "garbage"
    elif any(k in text for k in ["streetlight", "light", "lamp", "bulb"]):
        category = "streetlight"
    elif any(k in text for k in ["water", "leak", "pipe", "paani"]):
        category = "water_leakage"
    elif any(k in text for k in ["road", "sadak", "damage", "crack"]):
        category = "road_damage"

    severity = "medium"
    if any(k in text for k in ["urgent", "danger", "accident", "immediate"]):
        severity = "high"
    elif any(k in text for k in ["minor", "small", "little"]):
        severity = "low"

    return {
        "category": category,
        "severity": severity,
        "keywords": text.split()[:5],
    }