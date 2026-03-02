import json, pickle, os

DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
ASSESSMENTS_FILE = os.path.join(DATA_DIR, "shl_assessments.json")
META_FILE = os.path.join(DATA_DIR, "assessments_meta.pkl")

def build_doc(a):
    parts = []
    if a.get("name"): parts.append(f"Assessment: {a['name']}")
    if a.get("description"): parts.append(f"Description: {a['description']}")
    if a.get("test_type"): parts.append(f"Test Types: {', '.join(a['test_type'])}")
    if a.get("duration"): parts.append(f"Duration: {a['duration']} minutes")
    return " | ".join(parts)

def build_index():
    print("Building TF-IDF index...")
    with open(ASSESSMENTS_FILE) as f:
        assessments = json.load(f)
    print(f"Loaded {len(assessments)} assessments")
    documents = [build_doc(a) for a in assessments]
    meta = {"assessments": assessments, "documents": documents, "model_name": "tfidf"}
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved metadata -> {META_FILE}")
    print("✅ Done! (TF-IDF index will be built at API startup)")

if __name__ == "__main__":
    build_index()