"""
Generate Predictions CSV for Test Set
Calls the /recommend API for each test query and saves results in required format.

Usage:
  python generate_predictions.py --data ../data/Gen_AI_Dataset.xlsx --api http://localhost:8000
  
Output: predictions.csv
"""

import argparse
import pandas as pd
import requests
import csv
import os


def load_test_queries(path: str) -> list:
    df = pd.read_excel(path, sheet_name="Test-Set")
    return df["Query"].tolist()


def get_recommendations(api_base: str, query: str, top_k: int = 10) -> list:
    try:
        resp = requests.post(
            f"{api_base}/recommend",
            json={"query": query, "top_k": top_k},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return [a["url"] for a in data.get("recommended_assessments", [])]
    except Exception as e:
        print(f"  ERROR for query: {e}")
        return []


def generate_predictions(api_base: str, data_path: str, output_path: str = "predictions.csv"):
    queries = load_test_queries(data_path)
    print(f"Loaded {len(queries)} test queries")

    rows = []
    for i, query in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {query[:80]}...")
        urls = get_recommendations(api_base, query)
        print(f"  Got {len(urls)} recommendations")
        for url in urls:
            rows.append({"Query": query, "Assessment_url": url})

    df_out = pd.DataFrame(rows, columns=["Query", "Assessment_url"])
    df_out.to_csv(output_path, index=False)
    print(f"\n✅ Predictions saved → {output_path}")
    print(f"   Total rows: {len(rows)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/Gen_AI_Dataset.xlsx")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    generate_predictions(args.api, args.data, args.output)
