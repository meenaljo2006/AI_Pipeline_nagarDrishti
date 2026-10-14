from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

app = FastAPI(
    title="NagarDrishti AI Service",
    description="AI microservice for civic complaint image classification and NLP analysis",
    version="1.0.0",
)

# Allow Node.js backend to call this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": "NagarDrishti AI Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/health", "/analyze", "/classify-image", "/analyze-text"],
    }


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Service",
        "version": "1.0.0",
    }