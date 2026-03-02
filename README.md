# SHL Assessment Recommendation System

An intelligent, end-to-end recommendation engine that takes a natural language query or job description and returns the most relevant SHL Individual Test Solutions.

## Architecture

```
User Query / JD Text / URL
        │
        ▼
┌─────────────────────┐
│   Query Expansion   │  ← Gemini LLM extracts skills, domains, constraints
│   (Gemini Flash)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Semantic Retrieval │  ← FAISS + sentence-transformers (all-MiniLM-L6-v2)
│  Top-25 candidates  │     Cosine similarity over 377+ assessments
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Duration Filter    │  ← Removes assessments exceeding time constraints
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  LLM Re-ranking     │  ← Gemini ranks for relevance + balance
│  (Gemini Flash)     │     Ensures K/P/A test type balance
└─────────┬───────────┘
          │
          ▼
     Top 5-10 Results
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API key
```bash
export GEMINI_API_KEY="your-gemini-api-key"
# Get free key: https://ai.google.dev/gemini-api/docs/pricing
```

### 3. Scrape the SHL catalog
```bash
cd scraper
python scrape_catalog.py
# Output: data/shl_assessments.json (377+ assessments)
```

### 4. Build the vector index
```bash
cd backend
python build_index.py
# Output: data/faiss_index.bin + data/assessments_meta.pkl
```

### 5. Start the API
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Open the frontend
Open `frontend/index.html` in a browser, or serve it:
```bash
cd frontend
python -m http.server 3000
```

---

## API Reference

### Health Check
```
GET /health
Response: {"status": "healthy"}
```

### Recommend Assessments
```
POST /recommend
Content-Type: application/json

Body:
{
  "query": "I need to hire a Java developer who collaborates with business teams",
  "top_k": 10
}

Response:
{
  "recommended_assessments": [
    {
      "url": "https://www.shl.com/solutions/products/product-catalog/view/java-8-new/",
      "name": "Java 8 (New)",
      "adaptive_support": "No",
      "description": "...",
      "duration": 15,
      "remote_support": "Yes",
      "test_type": ["Knowledge & Skills"]
    },
    ...
  ]
}
```

---

## Evaluation

```bash
# Evaluate against labeled train set (requires API running)
cd evaluation
python evaluate.py --data ../data/Gen_AI_Dataset.xlsx --api http://localhost:8000

# Generate predictions for test set
python generate_predictions.py --data ../data/Gen_AI_Dataset.xlsx --api http://localhost:8000
# Output: predictions.csv
```

---

## Deployment (Free Tier)

### Option A: Render.com
1. Push code to GitHub
2. Create new Web Service on Render
3. Set `GEMINI_API_KEY` environment variable
4. Deploy — free tier gives 512MB RAM, sufficient for this app

### Option B: Railway.app
```bash
railway init
railway up
railway variables set GEMINI_API_KEY=your-key
```

### Option C: Docker
```bash
docker build -t shl-recommender .
docker run -e GEMINI_API_KEY=your-key -p 8000:8000 shl-recommender
```

---

## Project Structure

```
shl-recommendation/
├── scraper/
│   └── scrape_catalog.py       # SHL catalog scraper
├── backend/
│   ├── build_index.py          # FAISS index builder
│   ├── recommender.py          # Core RAG engine
│   └── main.py                 # FastAPI application
├── frontend/
│   └── index.html              # Web UI
├── evaluation/
│   ├── evaluate.py             # Mean Recall@10 evaluator
│   └── generate_predictions.py # Test set CSV generator
├── data/                       # (created after scraping)
│   ├── shl_assessments.json
│   ├── faiss_index.bin
│   └── assessments_meta.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Key Design Decisions

**Why FAISS + sentence-transformers?**  
Dense semantic retrieval captures meaning beyond keyword matching. `all-MiniLM-L6-v2` is fast (384-dim), production-ready, and runs on CPU.

**Why two LLM calls (expand + rerank)?**  
Query expansion improves recall by enriching sparse queries. Re-ranking improves precision and enforces balance across test types — crucial for the K/P balance requirement.

**Why Gemini Flash?**  
Free tier, fast, and more than sufficient for structured JSON extraction tasks.

**Duration filtering before LLM reranking?**  
Filtering first reduces context length for the LLM call and ensures constraint satisfaction.
