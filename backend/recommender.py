import os, json, pickle, re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
META_FILE = os.path.join(DATA_DIR, "assessments_meta.pkl")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-flash"
RETRIEVAL_TOP_K = 25
FINAL_TOP_K = 10

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement",
    "C": "Competencies", "D": "Development & 360", "E": "Assessment Exercises",
    "K": "Knowledge & Skills", "P": "Personality & Behavior", "S": "Simulations",
}

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine

class RecommendationEngine:
    def __init__(self):
        print("Loading assessment data...")
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        self.assessments = meta["assessments"]
        self.documents = meta["documents"]

        print("Building TF-IDF index...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.llm = genai.GenerativeModel(GEMINI_MODEL)
            print(f"Gemini {GEMINI_MODEL} ready")
        else:
            self.llm = None
        print(f"Engine ready. {len(self.assessments)} assessments indexed.")

    def _expand_query(self, query):
        if not self.llm:
            return query
        prompt = f"""You are an HR assessment expert. Given this hiring query, extract:
1. Technical skills (e.g. Python, Java, SQL)
2. Soft skills (e.g. communication, leadership)  
3. Cognitive requirements (e.g. numerical reasoning, verbal ability)
4. Job domain and seniority level

Return a concise 2-3 sentence enriched search query including assessment keywords like "knowledge test", "personality assessment", "aptitude test", "situational judgment".

Query: {query[:2000]}

Enriched query:"""
        try:
            return query[:500] + " " + self.llm.generate_content(prompt).text.strip()
        except:
            return query

    def _rerank_with_llm(self, query, candidates):
        if not self.llm or not candidates:
            return candidates
        candidate_list = "\n".join([
            f"{i+1}. [{a['name']}] Types: {', '.join(a.get('test_type',[]))} | "
            f"Duration: {a.get('duration','N/A')} min | {a.get('description','')[:120]}"
            for i, (a, _) in enumerate(candidates)
        ])
        duration_match = re.search(r"(\d+)\s*min", query, re.I)
        duration_note = f"Max duration: {duration_match.group(1)} minutes." if duration_match else ""
        prompt = f"""You are an SHL assessment consultant. Select and rank the BEST 5-10 assessments for this query.
RULES:
1. Must be relevant to the role and skills mentioned
2. {duration_note if duration_note else 'No duration limit.'}
3. Balance technical (Knowledge & Skills) AND behavioral (Personality & Behavior) if query needs both
4. Return ONLY a JSON array of 1-based indices e.g. [3,1,7,2,5]

QUERY: {query[:1500]}

CANDIDATES:
{candidate_list}

JSON array only:"""
        try:
            text = self.llm.generate_content(prompt).text.strip()
            match = re.search(r"\[[\d,\s]+\]", text)
            if match:
                indices = json.loads(match.group())
                seen, reranked = set(), []
                for idx in indices:
                    if 1 <= idx <= len(candidates) and idx not in seen:
                        seen.add(idx)
                        reranked.append(candidates[idx-1])
                return reranked[:FINAL_TOP_K]
        except Exception as e:
            print(f"Rerank failed: {e}")
        return candidates[:FINAL_TOP_K]

    def _duration_filter(self, candidates, query):
        m = re.search(r"(?:max|within|less than|under|at most)\s*(\d+)\s*min", query, re.I)
        if not m:
            m = re.search(r"(\d+)\s*min(?:utes?)?\s*(?:long|limit)?", query, re.I)
        if not m:
            return candidates
        max_d = int(m.group(1))
        filtered = [(a,s) for a,s in candidates if not a.get("duration") or a["duration"] <= max_d]
        return filtered if len(filtered) >= 5 else candidates

    def recommend(self, query, top_k=FINAL_TOP_K):
        expanded = self._expand_query(query)
        query_vec = self.vectorizer.transform([expanded])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        top_indices = scores.argsort()[::-1][:RETRIEVAL_TOP_K]
        candidates = [(self.assessments[i], float(scores[i])) for i in top_indices]
        candidates = self._duration_filter(candidates, query)
        final = self._rerank_with_llm(query, candidates)
        results = []
        for item in final[:top_k]:
            a = item[0] if isinstance(item, tuple) else item
            results.append({
                "name": a.get("name",""), "url": a.get("url",""),
                "description": a.get("description",""),
                "duration": a.get("duration"),
                "remote_support": a.get("remote_support","No"),
                "adaptive_support": a.get("adaptive_support","No"),
                "test_type": a.get("test_type",[]),
            })
        return results