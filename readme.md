# NagarDrishti AI Service

This is the Python-based AI microservice for the NagarDrishti platform. It uses **CLIP** (Computer Vision) for image classification and **Hugging Face Transformers** (NLP) for text analysis, sentiment, and severity scoring.

## 🚀 Setup & Installation

1. Clone the Repository
```bash
git clone https://github.com/meenaljo2006/AI_Pipeline_nagarDrishti.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a Virtual Environment
```bash
python -m venv venv
```
Activate it on Windows PowerShell:
```bash
.\venv\Scripts\Activate.ps1
```
When activated, the terminal usually shows (venv)

4. Running the service
```bash
uvicorn main:app --reload --port 8000
```

