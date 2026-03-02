"""
SHL Assessment Recommendation Engine
Uses FAISS for retrieval + Gemini for re-ranking and query understanding.
"""

import os
import json
import pickle
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
META_FILE = os.path.join(DATA_DIR, "assessments_meta.pkl")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"

RETRIEVAL_TOP_K = 25   # Retrieve top 25 from FAISS, re-rank to top 10
FINAL_TOP_K = 10

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# ── Singleton loader ──────────────────────────────────────────────────────────
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


# ── Engine ────────────────────────────────────────────────────────────────────
class RecommendationEngine:
    def __init__(self):
        print("Loading FAISS index and metadata...")
        self.index = faiss.read_index(INDEX_FILE)

        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)

        self.assessments = meta["assessments"]
        self.documents = meta["documents"]

        print(f"Loading embedding model: {meta['model_name']}...")
        self.embed_model = SentenceTransformer(meta["model_name"])

        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.llm = genai.GenerativeModel(GEMINI_MODEL)
            print(f"Gemini model: {GEMINI_MODEL} ✓")
        else:
            self.llm = None
            print("WARNING: GEMINI_API_KEY not set — running without LLM re-ranking")

        print(f"Engine ready. {len(self.assessments)} assessments indexed.")

    def _expand_query(self, query: str) -> str:
        """
        Use Gemini to extract key skills, test types, and duration constraints
        from the user query, returning an enriched search string.
        """
        if not self.llm:
            return query

        prompt = f"""You are an expert in HR assessments and psychometric testing.

Given this hiring query or job description, extract and list:
1. Technical skills required (e.g., Python, Java, SQL)
2. Soft skills required (e.g., communication, leadership, teamwork)
3. Cognitive/aptitude requirements (e.g., numerical reasoning, verbal ability)
4. Duration constraints (in minutes, if mentioned)
5. Seniority level (entry/mid/senior/graduate)
6. Job domain (e.g., sales, engineering, finance, customer service)

Return a concise enriched search query (2-4 sentences) that captures ALL these aspects for finding relevant psychometric assessments. Include relevant assessment keywords like "knowledge test", "personality assessment", "aptitude", "situational judgment" etc.

Query:
{query[:3000]}

Enriched search query:"""

        try:
            response = self.llm.generate_content(prompt)
            expanded = response.text.strip()
            # Combine original + expanded for better recall
            return f"{query[:500]} {expanded}"
        except Exception as e:
            print(f"Gemini query expansion failed: {e}")
            return query

    def _rerank_with_llm(self, query: str, candidates: list) -> list:
        """
        Use Gemini to re-rank candidates and ensure balance across test types.
        Returns ordered list of (assessment, score) tuples.
        """
        if not self.llm or not candidates:
            return candidates

        candidate_list = "\n".join([
            f"{i+1}. [{a['name']}] Types: {', '.join(a.get('test_type', []))} | "
            f"Duration: {a.get('duration', 'N/A')} min | {a.get('description', '')[:150]}"
            for i, (a, _) in enumerate(candidates)
        ])

        # Extract duration constraint from query
        duration_match = re.search(r"(\d+)\s*min", query, re.I)
        duration_note = f"Duration constraint: max {duration_match.group(1)} minutes." if duration_match else ""

        prompt = f"""You are an expert HR assessment consultant at SHL.

TASK: Select and rank the BEST 5-10 assessments from the candidates below for this hiring query.

RULES:
1. Assessments must be directly relevant to the role and skills mentioned.
2. {duration_note if duration_note else "No strict duration limit."}
3. Ensure BALANCE: if the query mentions both technical skills AND soft skills/personality, include assessments from BOTH domains (Knowledge & Skills type AND Personality & Behavior type).
4. Return ONLY a JSON array of numbers (1-based indices from the list below) in ranked order. Example: [3, 1, 7, 2, 5]

HIRING QUERY:
{query[:2000]}

CANDIDATE ASSESSMENTS:
{candidate_list}

Return ONLY the JSON array of indices (no explanation):"""

        try:
            response = self.llm.generate_content(prompt)
            text = response.text.strip()
            # Parse JSON array
            match = re.search(r"\[[\d,\s]+\]", text)
            if match:
                indices = json.loads(match.group())
                # Validate and deduplicate
                seen = set()
                reranked = []
                for idx in indices:
                    if 1 <= idx <= len(candidates) and idx not in seen:
                        seen.add(idx)
                        reranked.append(candidates[idx - 1])
                return reranked[:FINAL_TOP_K]
        except Exception as e:
            print(f"Gemini re-ranking failed: {e}")

        return candidates[:FINAL_TOP_K]

    def _apply_duration_filter(self, candidates: list, query: str) -> list:
        """Filter out assessments that exceed explicit duration constraints."""
        duration_match = re.search(r"(?:max|maximum|within|less than|under|at most)\s*(\d+)\s*min", query, re.I)
        if not duration_match:
            # Also check plain "X minutes"
            duration_match = re.search(r"(\d+)\s*min(?:utes?)?\s*(?:long|limit|cap)?", query, re.I)

        if not duration_match:
            return candidates

        max_duration = int(duration_match.group(1))
        filtered = [
            (a, s) for (a, s) in candidates
            if a.get("duration") is None or a["duration"] <= max_duration
        ]
        # If too few results after filtering, relax and return all
        if len(filtered) < 5:
            return candidates
        return filtered

    def recommend(self, query: str, top_k: int = FINAL_TOP_K) -> list:
        """
        Main recommendation pipeline:
        1. Query expansion via Gemini
        2. FAISS semantic retrieval (top 25)
        3. Duration filtering
        4. LLM re-ranking for relevance and balance
        """
        # Step 1: Expand query
        expanded_query = self._expand_query(query)

        # Step 2: Embed and retrieve
        query_embedding = self.embed_model.encode(
            [expanded_query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(query_embedding, RETRIEVAL_TOP_K)
        scores = scores[0]
        indices = indices[0]

        candidates = [
            (self.assessments[idx], float(scores[i]))
            for i, idx in enumerate(indices)
            if idx < len(self.assessments)
        ]

        # Step 3: Duration filter
        candidates = self._apply_duration_filter(candidates, query)

        # Step 4: LLM re-ranking
        final = self._rerank_with_llm(query, candidates)

        # Format output
        results = []
        for item in final[:top_k]:
            if isinstance(item, tuple):
                assessment, score = item
            else:
                assessment = item
            results.append({
                "name": assessment.get("name", ""),
                "url": assessment.get("url", ""),
                "description": assessment.get("description", ""),
                "duration": assessment.get("duration"),
                "remote_support": assessment.get("remote_support", "No"),
                "adaptive_support": assessment.get("adaptive_support", "No"),
                "test_type": assessment.get("test_type", []),
            })

        return results
