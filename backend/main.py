"""
SHL Assessment Recommendation API
FastAPI backend exposing /health and /recommend endpoints.

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from recommender import get_engine

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Recommends SHL Individual Test Solutions for a given job description or query.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10


class Assessment(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: list[str]


class RecommendResponse(BaseModel):
    recommended_assessments: list[Assessment]


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-load the engine so first request is fast."""
    get_engine()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    Recommend SHL assessments for a job description or natural language query.
    
    Accepts:
    - Natural language query
    - Full job description text
    - URL to a job description (fetches and parses automatically)
    
    Returns 5-10 most relevant Individual Test Solutions.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # If the query looks like a URL, fetch the content
    if query.startswith("http://") or query.startswith("https://"):
        try:
            resp = requests.get(query, timeout=10, headers={
                "User-Agent": "Mozilla/5.0"
            })
            resp.raise_for_status()
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            # Remove scripts/styles
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            query = soup.get_text(" ", strip=True)[:5000]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not fetch URL: {e}")

    top_k = max(1, min(request.top_k or 10, 10))

    try:
        engine = get_engine()
        results = engine.recommend(query, top_k=top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")

    if not results:
        raise HTTPException(status_code=404, detail="No assessments found for the given query.")

    return RecommendResponse(recommended_assessments=results)
