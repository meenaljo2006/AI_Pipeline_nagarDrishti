"""
NLP text analysis using Hugging Face transformers.
- Sentiment analysis with DistilBERT
- Zero-shot category classification
- Keyword extraction
"""
from transformers import pipeline
import re

print("Loading NLP models...")

# Sentiment analysis model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

# Zero-shot classification
zero_shot_pipeline = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

print("NLP models loaded")


# Civic issue category labels for zero-shot
CIVIC_LABELS = ["pothole", "garbage", "water leakage", "other civic issue"]

# Keywords that indicate urgency
URGENT_KEYWORDS = [
    "urgent", "emergency", "danger", "dangerous", "accident", "immediately",
    "critical", "severe", "hazard", "injury", "risk", "asap", "serious",
]

LOW_PRIORITY_KEYWORDS = [
    "minor", "small", "little", "not urgent", "slight", "tiny", "just",
]


def extract_keywords(text: str, max_keywords: int = 6) -> list:
    """Extract meaningful keywords from text (simple approach)"""
    # Remove punctuation, lowercase
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    words = cleaned.split()

    # Stopwords to remove
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "on", "in", "at",
        "to", "of", "and", "or", "for", "with", "by", "this", "that",
        "it", "be", "been", "has", "have", "had", "there", "here",
        "big", "very", "also", "my", "our", "your", "we", "i", "me",
    }

    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return keywords[:max_keywords]


def analyze_sentiment(text: str) -> dict:
    """Returns { label: 'POSITIVE'|'NEGATIVE', score: 0.0-1.0 }"""
    if not text or not text.strip():
        return {"label": "NEUTRAL", "score": 0.0}

    try:
        result = sentiment_pipeline(text[:512])[0]
        return {
            "label": result["label"],
            "score": round(result["score"], 3),
        }
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"label": "NEUTRAL", "score": 0.0}


def classify_text_category(text: str) -> dict:
    """Zero-shot category classification from text"""
    if not text or not text.strip():
        return {"category": "other", "confidence": 0.0, "all_scores": {}}

    try:
        result = zero_shot_pipeline(text[:512], CIVIC_LABELS)
        # Map label "water leakage" → "water_leakage"
        top_label = result["labels"][0].replace(" ", "_")
        if top_label == "other_civic_issue":
            top_label = "other"

        all_scores = {
            label.replace(" ", "_").replace("other_civic_issue", "other"): round(score, 3)
            for label, score in zip(result["labels"], result["scores"])
        }

        return {
            "category": top_label,
            "confidence": round(result["scores"][0], 3),
            "all_scores": all_scores,
        }
    except Exception as e:
        print(f"Zero-shot error: {e}")
        return {"category": "other", "confidence": 0.0, "all_scores": {}}


def calculate_urgency(text: str, sentiment: dict) -> float:
    """Calculate urgency score 0.0 to 1.0"""
    text_lower = text.lower()
    urgency = 0.5  # baseline

    # Boost for urgent keywords
    for kw in URGENT_KEYWORDS:
        if kw in text_lower:
            urgency += 0.15

    # Reduce for "minor" keywords
    for kw in LOW_PRIORITY_KEYWORDS:
        if kw in text_lower:
            urgency -= 0.15

    # Boost if very negative sentiment
    if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.9:
        urgency += 0.1

    return round(max(0.0, min(1.0, urgency)), 2)


def severity_from_urgency(urgency: float) -> str:
    """Convert urgency score to severity label"""
    if urgency >= 0.7:
        return "high"
    elif urgency >= 0.4:
        return "medium"
    return "low"


def analyze_text(description: str) -> dict:
    """
    Full NLP analysis of complaint text.
    Returns category, sentiment, keywords, urgency, severity.
    """
    if not description or not description.strip():
        return {
            "category": "other",
            "confidence": 0.0,
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "keywords": [],
            "urgency_score": 0.5,
            "severity": "medium",
        }

    # Run all analyses
    sentiment = analyze_sentiment(description)
    text_category = classify_text_category(description)
    keywords = extract_keywords(description)
    urgency = calculate_urgency(description, sentiment)
    severity = severity_from_urgency(urgency)

    return {
        "category": text_category["category"],
        "confidence": text_category["confidence"],
        "all_scores": text_category["all_scores"],
        "sentiment": sentiment["label"].lower(),
        "sentiment_score": sentiment["score"],
        "keywords": keywords,
        "urgency_score": urgency,
        "severity": severity,
    }