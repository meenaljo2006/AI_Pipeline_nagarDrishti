"""
Image classification using CLIP zero-shot.
Categorizes civic issue photos without any training.
"""
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO

# Load CLIP model once at startup (cached in memory)
print("Loading CLIP model...")
MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("CLIP model loaded")


# Category descriptions — how CLIP will match the image
CATEGORY_PROMPTS = {
    "pothole": [
        "a photo of a pothole on a road",
        "a photo of a hole in the road",
        "a photo of damaged asphalt",
    ],
    "garbage": [
        "a photo of a garbage pile on the street",
        "a photo of trash on the ground",
        "a photo of waste dumped on a road",
    ],
    "water_leakage": [
        "a photo of water leaking from a pipe",
        "a photo of a water leak on the street",
        "a photo of a burst water pipe",
    ],
    "other": [
        "a photo of a civic issue",
        "a photo of a public infrastructure problem",
    ],
}


def download_image(image_url: str) -> Image.Image:
    """Download image from URL"""
    headers = {"User-Agent": "NagarDrishti/1.0"}
    response = requests.get(image_url, headers=headers, timeout=15)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def classify_image(image_url: str) -> dict:
    """
    Classify a civic issue image into a category.

    Returns:
        {
            "category": "pothole",
            "confidence": 0.87,
            "all_scores": {"pothole": 0.87, "garbage": 0.05, ...}
        }
    """
    try:
        # Download image
        image = download_image(image_url)
        print(f"📸 Downloaded image: {image.size}")

        # Flatten prompts into list
        all_prompts = []
        prompt_to_category = []
        for category, prompts in CATEGORY_PROMPTS.items():
            for p in prompts:
                all_prompts.append(p)
                prompt_to_category.append(category)

        # Run CLIP
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        outputs = model(**inputs)

        # Get probabilities
        probs = outputs.logits_per_image.softmax(dim=1)[0]

        # Aggregate by category 
        scores = {}
        for prompt_idx, category in enumerate(prompt_to_category):
            score = float(probs[prompt_idx])
            if category not in scores or score > scores[category]:
                scores[category] = score

        # Find best category
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        print(f"Classified: {best_category} ({best_score:.2f})")

        return {
            "category": best_category,
            "confidence": round(best_score, 3),
            "all_scores": {k: round(v, 3) for k, v in scores.items()},
        }

    except Exception as e:
        print(f"Classification error: {e}")
        return {
            "category": "other",
            "confidence": 0.0,
            "all_scores": {},
            "error": str(e),
        }