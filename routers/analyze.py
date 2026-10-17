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
    if not req.imageUrl:
        raise HTTPException(status_code=400, detail="imageUrl is required")
    return classify_image(req.imageUrl)


@router.post("/analyze-text")
async def analyze_text_endpoint(req: TextRequest):
    if not req.description:
        raise HTTPException(status_code=400, detail="description is required")
    return analyze_text(req.description)


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
    Combined analysis: Image + Text + Cross-Validation + Severity.
    """
    # 1. Run Image Classification
    image_result = classify_image(req.imageUrl)
    img_cat = image_result["category"]
    img_conf = image_result["confidence"]

    # 2. Run Text Analysis
    text_result = analyze_text(req.description)
    txt_cat = text_result["category"]
    txt_conf = text_result["confidence"]
    txt_urgency = text_result["urgency_score"]

    # 3. CROSS-VALIDATION LOGIC
    is_valid = True
    rejection_reason = None
    final_category = img_cat

    if img_cat == "other" and txt_cat != "other":
        # Image is unclear, but text is clear. Trust text.
        final_category = txt_cat
    elif img_cat != "other" and txt_cat == "other":
        # Image is clear, text is vague. Trust image.
        final_category = img_cat
    elif img_cat != "other" and txt_cat != "other":
        # Both are clear. Check for mismatch.
        if img_cat != txt_cat:
            is_valid = False
            rejection_reason = f"Image shows '{img_cat}' but text says '{txt_cat}'."
            final_category = img_cat # Default to image category
    elif img_cat == "other" and txt_cat == "other":
        # Neither is clear
        is_valid = False
        rejection_reason = "Could not identify the issue from image or text."

    # 4. SEVERITY ASSESSMENT (Only if valid)
    severity = "low"
    severity_score = 0.0
    baseline = 0.0 # Default if invalid

    if is_valid:
        baseline = CATEGORY_BASELINE.get(final_category, 0.4)
        
        # Weighted Score: 50% Category + 30% Text Urgency + 20% Image Confidence
        severity_score = (
            (0.5 * baseline) + 
            (0.3 * txt_urgency) + 
            (0.2 * img_conf)
        )
        
        if severity_score >= 0.65:
            severity = "high"
        elif severity_score >= 0.4:
            severity = "medium"
        else:
            severity = "low"

    # --- BEAUTIFUL LOGGING SECTION ---
    print("\n---------------------------------------------------------")
    print("STEP 1: AI PIPELINE STARTED (Python Microservice)")
    print("---------------------------------------------------------")
    print(f"[CLIP MODEL] Detected Image Category: {img_cat} (Confidence: {img_conf})")
    print(f"[NLP MODEL]  Detected Text Category: {txt_cat} (Confidence: {txt_conf})")

    print("\n---------------------------------------------------------")
    print("STEP 2: CROSS-VALIDATION")
    print("---------------------------------------------------------")
    if is_valid:
        print(f"Match! Image and Text align on '{final_category}'. Validating...")
    else:
        print(f"Mismatch! {rejection_reason}")

    if is_valid:
        print("\n---------------------------------------------------------")
        print("STEP 3: SEVERITY ASSESSMENT")
        print("---------------------------------------------------------")
        print(f"Urgency Score: {txt_urgency} | Category Baseline: {baseline}")
        print(f"Final Severity: {severity.upper()} (Score: {severity_score:.2f})")

    # 5. Build Final Response
    result = {
        "isValid": is_valid,
        "rejectionReason": rejection_reason,
        "category": final_category,
        "imageCategory": img_cat,
        "textCategory": txt_cat,
        "imageConfidence": img_conf,
        "keywords": text_result["keywords"],
        "sentiment": text_result["sentiment"],
        "urgencyScore": txt_urgency,
        "severityScore": round(severity_score, 3),
        "severity": severity,
    }
    
    return result