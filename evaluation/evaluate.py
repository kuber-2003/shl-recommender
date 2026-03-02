import argparse, pandas as pd, requests
from collections import defaultdict

def norm(url):
    url = url.strip().rstrip("/")
    url = url.replace("https://www.shl.com/products/", "https://www.shl.com/solutions/products/")
    return url.lower()

def load_train(path):
    df = pd.read_excel(path, sheet_name="Train-Set")
    q2u = defaultdict(list)
    for _, row in df.iterrows():
        q2u[row["Query"]].append(norm(row["Assessment_url"]))
    return dict(q2u)

def recall_k(rec, rel, k=10):
    r = set(norm(u) for u in rec[:k])
    g = set(norm(u) for u in rel)
    return len(r & g) / len(g) if g else 0.0

def run(api, q2u, k=10):
    recalls = []
    for q, rel in q2u.items():
        print(f"Query: {q[:70]}...")
        try:
            resp = requests.post(f"{api}/recommend", json={"query": q, "top_k": k}, timeout=60)
            rec = [a["url"] for a in resp.json().get("recommended_assessments", [])]
        except Exception as e:
            print(f"  ERROR: {e}"); rec = []
        r = recall_k(rec, rel, k)
        recalls.append(r)
        print(f"  Recall@{k}: {r:.3f}  ({len(rec)} recommended, {len(rel)} relevant)")
        matched = set(norm(u) for u in rec) & set(norm(u) for u in rel)
        if matched:
            print(f"  MATCHED: {len(matched)} assessments")
    mean = sum(recalls) / len(recalls) if recalls else 0
    print(f"\nMean Recall@{k}: {mean:.4f}")
    print(f"Scores: {[round(r, 3) for r in recalls]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    run(a.api, load_train(a.data), a.k)