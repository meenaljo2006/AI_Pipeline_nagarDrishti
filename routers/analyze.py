from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.classifier import classify_image
from services.nlp import analyze_text

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
    Full NLP analysis: sentiment + zero-shot category + urgency + severity.
    """
    if not req.description:
        raise HTTPException(status_code=400, detail="description is required")

    result = analyze_text(req.description)
    print(f"NLP: {result['category']} | sentiment: {result['sentiment']} | severity: {result['severity']}")
    return result