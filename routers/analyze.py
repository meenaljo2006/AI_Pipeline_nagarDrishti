from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.classifier import classify_image
from services.nlp import analyze_text

router = APIRouter()


class ImageRequest(BaseModel):
    imageUrl: str

class TextRequest(BaseModel):
    description: str

class AnalyzeRequest(BaseModel):
    imageUrl: str
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

# Category baseline severities
CATEGORY_BASELINE = {
    "water_leakage": 0.7,
    "pothole": 0.5,
    "garbage": 0.35,
    "other": 0.4,
}

@router.post("/analyze")
async def analyze_complaint(req: AnalyzeRequest):
    """
    Combined analysis: image classification + NLP + severity assessment.
    Returns everything the Node.js backend needs.
    """
    # 1. Image classification
    image_result = classify_image(req.imageUrl)

    # 2. Text analysis
    text_result = analyze_text(req.description)

    # 3. Severity assessment — combine image category + text urgency
    category = image_result["category"]
    baseline = CATEGORY_BASELINE.get(category, 0.4)
    text_urgency = text_result["urgency_score"]
    image_confidence = image_result["confidence"]

    # Weighted severity score (0.0 - 1.0)
    # 50% category baseline + 30% text urgency + 20% image confidence
    severity_score = (
        0.5 * baseline +
        0.3 * text_urgency +
        0.2 * image_confidence
    )

    # Convert to label
    if severity_score >= 0.65:
        severity = "high"
    elif severity_score >= 0.4:
        severity = "medium"
    else:
        severity = "low"

    # Determine if valid (complaint is real)
    is_valid = (
        image_result["confidence"] >= 0.35 and
        category != "other"
    )

    result = {
        "isValid": is_valid,
        "category": category,
        "imageConfidence": image_result["confidence"],
        "imageScores": image_result.get("all_scores", {}),
        "textCategory": text_result["category"],
        "textConfidence": text_result["confidence"],
        "keywords": text_result["keywords"],
        "sentiment": text_result["sentiment"],
        "sentimentScore": text_result["sentiment_score"],
        "urgencyScore": text_result["urgency_score"],
        "severityScore": round(severity_score, 3),
        "severity": severity,
    }

    print(f"ANALYZE: {category} | severity: {severity} ({severity_score:.2f}) | valid: {is_valid}")
    return result