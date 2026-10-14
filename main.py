from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from routers import analyze

app = FastAPI(
    title="NagarDrishti AI Service",
    description="AI microservice for civic complaint analysis",
    version="1.0.0",
)

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
    }


@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Service",
        "version": "1.0.0",
    }


# Register routers
app.include_router(analyze.router, tags=["Analysis"])