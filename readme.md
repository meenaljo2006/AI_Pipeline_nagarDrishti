# NagarDrishti AI Service

The AI service is the Python-based analysis component of **NagarDrishti
--- See. Verify. Resolve.** It exposes HTTP endpoints that help analyze
citizen-submitted civic complaints using image classification, text
analysis, and custom validation/severity rules.

## Overview

NagarDrishti is a civic-issue reporting platform for issues such as
potholes, garbage accumulation, water leakage, waterlogging, and damaged
street infrastructure. The AI service helps analyze the evidence
supplied with a complaint.

Its intended responsibilities are:

1.  **Image classification:** use CLIP zero-shot image classification to
    compare a submitted image with text descriptions of supported civic
    categories.
2.  **Text analysis:** analyze the citizen's description. Depending on
    the implementation, this may include sentiment analysis using a
    Hugging Face Transformer, category detection, keyword extraction,
    and urgency scoring.
3.  **Cross-validation:** compare the image-based category with the
    text-based category and identify possible mismatches.
4.  **Severity assessment:** apply documented rules to assign a low,
    medium, or high severity level.
5.  **HTTP API:** expose the analysis functions through FastAPI so the
    main backend can call them.

The AI service provides analysis signals; it does not replace human
verification or municipal decision-making.

## Architecture and request flow

``` text
Citizen submits a photo and description
                 |
                 v
        NagarDrishti backend
                 |
                 v
       Python FastAPI AI service
                 |
         +-------+--------+
         |                |
         v                v
  CLIP image analysis   Text analysis
         |                |
         +-------+--------+
                 |
                 v
     Cross-validation and rules
                 |
                 v
 category + confidence + severity
 + validation result (as implemented)
                 |
                 v
        Main backend / dashboard
```


### Main Components

| Component | Responsibility |
|---|---|
| **FastAPI** | Defines HTTP routes and validates API requests. |
| **CLIP** | Compares an image with candidate text labels and ranks likely categories. |
| **Hugging Face Transformers** | Loads and runs pre-trained NLP models where configured. |
| **Custom Python Rules** | Extracts configured keywords and calculates urgency and severity where implemented. |
| **Node.js Backend** | Manages the main application workflow and calls the AI service. |

## Technology stack

-   **Python 3.10--3.12** --- runtime (confirm compatibility with the
    installed dependency versions).
-   **FastAPI** --- HTTP API framework.
-   **Uvicorn** --- ASGI development server.
-   **Transformers** --- model loading and inference.
-   **CLIP** --- zero-shot image classification.
-   **PyTorch** --- model execution dependency, normally installed with
    the selected Transformers/PyTorch setup.
-   **Pydantic** --- request and response validation through FastAPI.
-   **Hugging Face pre-trained models** --- model weights are downloaded
    on first use and typically cached locally.

The exact package versions should be pinned in `requirements.txt` for
reproducible installation.

## Setup and installation

Run these commands from the repository root.

### 1. Clone the repository

``` bash
git clone https://github.com/meenaljo2006/AI_Pipeline_nagarDrishti.git
```

### 2. Create a virtual environment

``` bash
python -m venv venv
```

Activate it on Windows PowerShell:

``` powershell
.\venv\Scripts\Activate.ps1
```

When activated, the terminal usually shows `(venv)`.

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Verify key imports

``` bash
python -c "from transformers import CLIPProcessor, CLIPModel; print('CLIP imports successfully')"
```

## Configuration and model downloads
NagarDrishti's AI service uses three pre-trained models from Hugging Face to perform image classification and natural language processing (NLP). These models are loaded when their respective Python modules are initialized and reused for subsequent predictions.

The project does not train these models from scratch. Instead, it uses pre-trained model weights and applies them to civic complaint analysis.

The service may load model weights the first time a model is used. This
can require a substantial download and may take longer than later
requests. Ensure there is sufficient disk space and a stable internet
connection for initial setup.

### 1. Models Used

| Model | Model Identifier | Purpose |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | Performs zero-shot image classification by comparing an image with candidate text descriptions of civic issues. |
| DistilBERT Sentiment Model | `distilbert-base-uncased-finetuned-sst-2-english` | Analyzes the sentiment of a complaint and returns a positive or negative label with a confidence score. |
| DistilBERT-MNLI | `typeform/distilbert-base-uncased-mnli` | Performs zero-shot text classification to identify the civic issue described by the citizen. |

### 2. How Model Loading Works

The service loads the models using the Hugging Face `transformers` library.

**Image classification (CLIP):**

The following components are initialized in the image-classification module:

- `CLIPModel.from_pretrained()` loads the pre-trained CLIP model.
- `CLIPProcessor.from_pretrained()` loads the processor required to prepare images and text for the model.

**Text analysis (NLP):**

The text-analysis module initializes two pipelines:

- `pipeline("sentiment-analysis", ...)` loads the sentiment analysis model.
- `pipeline("zero-shot-classification", ...)` loads the zero-shot text classification model.

These models are initialized at module startup rather than being reloaded for every complaint. This allows the application to reuse the loaded models when processing subsequent requests.

## Running the service

With the virtual environment active:

``` bash
uvicorn main:app --reload --port 8000
```

-   Local base URL: `http://localhost:8000`
-   Interactive Swagger documentation: `http://localhost:8000/docs`

## API reference

The following endpoint list reflects the proposed API in the project
notes. Confirm each route and its schema in `main.py` and `/docs`.

### 1. `GET /health` --- health check

**Purpose:** check whether the service is responding.

Example:

``` bash
http://localhost:8000/health
```
A basic health endpoint may return a status such as `{"status":"ok"}`. 

### 2. `POST /classify-image` --- image category prediction

**Purpose:** classify an image into one of the candidate civic
categories configured in the application.

The project describes an `imageUrl` field.
``` json
{
  "imageUrl": "https://hips.hearstapps.com/hmg-prod/images/5-things-you-need-to-know-about-potholes-1674217331.jpg"
}
```

Response fields, depending on implementation:

``` json
{
  "category": "pothole",
  "confidence": 0.636,
  "all_scores": {
    "pothole": 0.636,
    "garbage": 0,
    "water_leakage": 0.002,
    "other": 0.016
  }
}
```

### 3. `POST /analyze-text` --- description analysis

**Purpose:** analyze a citizen's text description.

Request:

``` json
{
  "description": "A large pothole on the main road is causing accidents."
}
```

Depending on the implemented pipeline, the response contain a text
category, sentiment, extracted keywords, urgency score, and severity.
For example:

``` json
{
  "category": "pothole",
  "confidence": 0.985,
  "all_scores": {
    "pothole": 0.985,
    "other": 0.007,
    "water_leakage": 0.006,
    "garbage": 0.002
  },
  "sentiment": "negative",
  "sentiment_score": 1,
  "keywords": [
    "large",
    "pothole",
    "main",
    "road",
    "causing",
    "accidents"
  ],
  "urgency_score": 0.75,
  "severity": "high"
}
```

### 4. `POST /analyze` --- combined complaint analysis

**Purpose:** combine image and description analysis, cross-validate the
category, and calculate a final validation/severity result according to
the implemented rules.

Request:

``` json
{
  "imageUrl": "https://example.com/civic-issue.jpg",
  "description": "A large pothole on the main road is causing accidents."
}
```

Successful response could look like:

``` json
{
  "isValid": true,
  "rejectionReason": null,
  "category": "pothole",
  "imageCategory": "pothole",
  "textCategory": "pothole",
  "imageConfidence": 0.636,
  "keywords": [
    "large",
    "pothole",
    "main",
    "road",
    "causing",
    "accidents"
  ],
  "sentiment": "negative",
  "urgencyScore": 0.75,
  "severityScore": 0.602,
  "severity": "medium"
}
```

A mismatch response could communicate that the image prediction and
description disagree. 

## How the analysis works
The service does not train these models from scratch. It loads pre-trained models and combines their predictions with custom Python logic.

### 1. CLIP image classification

**Model:** `openai/clip-vit-base-patch32`

**Libraries:** Hugging Face Transformers, PyTorch, Pillow, Requests

CLIP (Contrastive Language–Image Pre-training) is a vision-language model that compares images with text descriptions. NagarDrishti uses CLIP's zero-shot classification capability to identify the civic issue that best matches a submitted photograph.

#### How It Works

1. The service receives an image URL.
2. The image is downloaded using the Python Requests library.
3. Pillow converts the downloaded image into RGB format.
4. CLIPProcessor prepares the image and candidate text descriptions for the model.
5. CLIPModel calculates image–text similarity scores.
6. Softmax converts the image's text-prompt scores into normalized scores across all candidate prompts.
7. The application groups prompts by civic category and selects the highest-scoring category.

#### Supported Image Categories

The current implementation defines the following categories and candidate descriptions:

| Category | Example Candidate Descriptions |
|---|---|
| `pothole` | A photo of a pothole; a damaged road with a deep hole. |
| `garbage` | A photo of a garbage pile; trash bags and waste on the ground. |
| `water_leakage` | Water leaking from a pipe; water spilling onto the street from a broken pipe. |
| `other` | A civic infrastructure problem; a public facility issue. |

Each category has multiple candidate descriptions to provide the model with different ways of matching the image.

### 2. Text analysis

The text analysis module uses two pre-trained Hugging Face Transformer models and custom Python functions to understand a citizen's complaint.

#### 2.1 Sentiment Analysis

**Model:** `distilbert-base-uncased-finetuned-sst-2-english`

The sentiment model predicts whether the complaint text has positive or negative sentiment.

The `analyze_sentiment()` function returns:

- `label`: `POSITIVE` or `NEGATIVE` when inference succeeds.
- `score`: the model's confidence score, rounded to three decimal places.

The sentiment score measures the model's confidence in its sentiment prediction; it does not directly measure the physical danger or severity of a civic issue.

#### 2.2 Zero-Shot Text Classification

**Model:** `typeform/distilbert-base-uncased-mnli`

The zero-shot classification model predicts which configured civic category best matches the citizen's description without requiring a separately trained classifier for each category.

The function returns the predicted category, its score, and the scores for all candidate labels.

#### 2.3 Keyword Extraction

The `extract_keywords()` function uses a simple rule-based approach rather than a separate machine-learning model.

The process is:

1. Convert the description to lowercase.
2. Remove punctuation.
3. Split the text into words.
4. Remove configured stopwords and words with two or fewer characters.
5. Return up to six remaining words by default.

The actual output depends on the input text and the configured stopword list. This method does not rank keywords by importance or perform advanced entity extraction.

#### 2.4 Urgency Scoring

The `calculate_urgency()` function calculates a rule-based urgency score between `0.0` and `1.0`.

The initial score is `0.5`.

| Condition | Score Adjustment |
|---|:---:|
| Each configured urgent keyword found | +0.15 |
| Each configured low-priority keyword found | -0.15 |
| Negative sentiment with a sentiment score greater than 0.9 | +0.10 |
| Final score | Clamped between 0.0 and 1.0 |

Urgency keywords include terms such as `urgent`, `emergency`, `danger`, `accident`, `hazard`, and `injury`.

Low-priority keywords include `minor`, `small`, `slight`, and `tiny`.

The implementation checks whether each keyword appears as a substring in the lowercased description. It does not perform advanced contextual or negation analysis. For example, the phrase `not dangerous` may still trigger the `dangerous` keyword rule.

#### 2.5 Text-Based Severity

The standalone NLP function `severity_from_urgency()` converts the urgency score into a severity label.

| Urgency Score | Severity |
|---|---|
| Greater than or equal to `0.7` | `high` |
| Greater than or equal to `0.4` and below `0.7` | `medium` |
| Below `0.4` | `low` |

The `/analyze-text` endpoint returns the category, confidence, all category scores, sentiment, sentiment score, extracted keywords, urgency score, and text-based severity.

### 3. Cross-validation

The `/analyze` endpoint combines the output of CLIP image classification and NLP text analysis to check whether both sources identify a consistent civic issue.
The endpoint compares the image category with the text category.

#### Validation Rules

| Image Category | Text Category | Result |
|---|---|---|
| `other` | A specific category | Invalid; reject the complaint because the image does not clearly identify a supported civic issue. |
| `other` | `other` | Invalid; reject the complaint because the image does not clearly identify a supported civic issue. |
| A specific category | `other` | Valid; use the image category because the image provides clearer evidence than the vague text description. |
| Same specific category | Same category | Valid; accept the complaint because the image and text categories match. |
| Different specific categories | Different categories | Invalid; reject the complaint and record a mismatch reason describing the disagreement between the image and text categories. |

### 4. Combined Severity Assessment

For valid complaints, the `/analyze` endpoint calculates a combined severity score using three signals:

1. Category baseline severity.
2. Text-based urgency score.
3. Image classification confidence.

#### Category Baseline Scores

The following baseline values are configured in the FastAPI routes:

| Category | Baseline Score |
|---|---:|
| `water_leakage` | 0.70 |
| `pothole` | 0.50 |
| `garbage` | 0.35 |
| `other` | 0.40 |

#### Severity Formula

The combined severity score is calculated as:

```text
Severity Score =
    (0.50 × Category Baseline)
  + (0.30 × Text Urgency)
  + (0.20 × Image Confidence)
```

This means that the category baseline contributes 50% of the score, text urgency contributes 30%, and image confidence contributes 20%.

#### Final Severity Thresholds

| Combined Severity Score | Final Severity |
|---|---|
| Greater than or equal to `0.65` | `high` |
| Greater than or equal to `0.40` and below `0.65` | `medium` |
| Below `0.40` | `low` |

If the complaint is invalid, the current implementation assigns a severity of `low` and a severity score of `0.0`.

**Note:** Combined severity is calculated only for valid complaints. It is separate from the standalone text severity returned by `/analyze-text`.


### 5. Duplicate detection (separate workflow)

Duplicate detection is a separate feature from the current image and text analysis pipeline.

The proposed workflow is to identify likely duplicate complaints by comparing their category, geographic proximity, reporting time, and status.

A proposed rule is to search for complaints in the same category within 100 metres of the reported location over the previous seven days, excluding complaints that are resolved, closed, or rejected.

MongoDB geospatial indexes and queries can support this workflow in the main backend, where complaint records and location data are maintained.

**Implementation status:** Duplicate detection is not implemented in the FastAPI routes or Python modules shown in this documentation. It should be documented as planned work until the database query, duplicate linkage, and associated tests are implemented.

## Integration with the Node.js backend

The main backend should send an HTTP request to the AI service and
handle its response. For local development, the AI service might run on
port `8000`; the backend should use a configurable base URL rather than
scattering the URL throughout the code.

Illustrative Node.js example using `fetch`:

``` javascript
const AI_SERVICE_URL =
  process.env.AI_SERVICE_URL || "http://localhost:8000";

async function analyzeComplaint(imageUrl, description) {
  const response = await fetch(`${AI_SERVICE_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ imageUrl, description })
  });

  if (!response.ok) {
    throw new Error(`AI service returned HTTP ${response.status}`);
  }

  return await response.json();
}
```

This example shows the endpoint accepts `imageUrl` and `description`.
Adjust the payload to the actual Pydantic schema. Add controlled error handling in production.

------------------------------------------------------------------------

**Project:** NagarDrishti --- See. Verify. Resolve.\
**Service:** Python AI microservice\
**Development server:** `http://localhost:8000`\
**Interactive API documentation:** `http://localhost:8000/docs`
