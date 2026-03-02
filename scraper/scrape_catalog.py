"""
SHL Assessment Catalog Scraper - Fixed Version
Run from ANY directory. Creates data/ folder automatically.
"""
import requests
from bs4 import BeautifulSoup
import json, time, re, os
from urllib.parse import urljoin

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(DATA_DIR, "shl_assessments.json")

BASE_URL = "https://www.shl.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
TEST_TYPE_MAP = {
    "A": "Ability & Aptitude", "B": "Biodata & Situational Judgement",
    "C": "Competencies", "D": "Development & 360",
    "E": "Assessment Exercises", "K": "Knowledge & Skills",
    "P": "Personality & Behavior", "S": "Simulations",
}

def fetch_catalog_urls(session):
    all_urls = set()
    # Try both known catalog URL patterns
    for base in [
        "https://www.shl.com/solutions/products/product-catalog/",
        "https://www.shl.com/products/product-catalog/",
    ]:
        for start in range(0, 5000, 12):
            url = f"{base}?start={start}&type=1"
            try:
                resp = session.get(url, timeout=15)
                soup = BeautifulSoup(resp.text, "html.parser")
                links = {urljoin(BASE_URL, a["href"]) for a in soup.find_all("a", href=True)
                         if "product-catalog/view" in a.get("href", "")}
                if not links and start > 0:
                    break
                all_urls.update(links)
                print(f"  {base.split('//')[1][:30]} start={start}: +{len(links)} = {len(all_urls)} total")
                time.sleep(0.4)
            except Exception as e:
                print(f"  Error: {e}")
                break
        if len(all_urls) > 50:
            break
    return list(all_urls)

def parse_page(url, session):
    try:
        r = session.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        # Name
        h1 = soup.find("h1")
        name = h1.get_text(strip=True) if h1 else url.rstrip("/").split("/")[-1].replace("-", " ").title()

        # Description - meta is most reliable
        desc = ""
        meta = soup.find("meta", {"name": "description"})
        if meta:
            desc = meta.get("content", "")
        if not desc:
            p = soup.find("p")
            desc = p.get_text(strip=True) if p else ""

        # Test types from badges and text
        types = []
        for el in soup.find_all(True):
            t = el.get_text(strip=True).upper()
            if t in TEST_TYPE_MAP and TEST_TYPE_MAP[t] not in types:
                types.append(TEST_TYPE_MAP[t])
        m = re.search(r"Test Type[:\s]+([A-Z\s]{1,20})", text)
        if m:
            for c in m.group(1).split():
                if c in TEST_TYPE_MAP and TEST_TYPE_MAP[c] not in types:
                    types.append(TEST_TYPE_MAP[c])

        # Duration
        duration = None
        for pat in [r"Approximate Completion Time in minutes\s*[=:]\s*(\d+)", r"(\d+)\s*min"]:
            m2 = re.search(pat, text, re.I)
            if m2:
                duration = int(m2.group(1))
                break

        remote = "Yes" if "Remote Testing" in text else "No"
        adaptive = "Yes" if re.search(r"Adaptive\s*(Testing|Support)", text, re.I) else "No"

        return {"name": name, "url": url, "description": desc[:500],
                "test_type": types[:5], "duration": duration,
                "remote_support": remote, "adaptive_support": adaptive}
    except Exception as e:
        print(f"    ERROR: {e}")
        return None

def scrape_all():
    print("="*60)
    print("SHL Catalog Scraper")
    print(f"Output: {OUTPUT_FILE}")
    print("="*60)
    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n[1/2] Collecting URLs...")
    urls = fetch_catalog_urls(session)
    print(f"\nFound {len(urls)} URLs")

    if len(urls) < 100:
        print("\n*** NOTE: SHL's catalog uses JavaScript to render items.")
        print("*** The static scraper only gets partial results.")
        print("*** See instructions below for the Selenium scraper.\n")

    print("\n[2/2] Scraping pages...")
    results = []
    for i, url in enumerate(urls):
        slug = url.rstrip("/").split("/")[-1]
        print(f"  [{i+1}/{len(urls)}] {slug}")
        data = parse_page(url, session)
        if data:
            results.append(data)
        time.sleep(0.25)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Done. Scraped {len(results)} assessments → {OUTPUT_FILE}")
    return results

if __name__ == "__main__":
    scrape_all()