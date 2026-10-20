"""Fetch primary paper copies for this read-only literature audit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "agarwal_2025": "https://shendure-web.gs.washington.edu/documents/Agarwal_Nature_2025.pdf",
    "inoue_2017": "https://genome.cshlp.org/content/genome/27/1/38.full.pdf",
    "klein_2020": "https://krishna.gs.washington.edu/documents/klein_nature_methods_2020.pdf",
    "kreimer_2019": "https://escholarship.org/content/qt2tn3m9xk/qt2tn3m9xk.pdf",
    "rafi_2025_preprint": "https://www.biorxiv.org/content/10.1101/2025.01.22.634321v1.full.pdf",
    "sefer_2022": "https://bmcgenomics.biomedcentral.com/counter/pdf/10.1186/s12864-022-08498-5.pdf",
}

def fetch(item):
    name, url = item
    result = {"id": name, "url": url}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Academic research source verification"})
        with urllib.request.urlopen(request, timeout=40) as response:
            data = response.read()
            result["resolved_url"] = response.url
        if not data.startswith(b"%PDF"):
            raise ValueError("The response was not a PDF")
        target = ROOT / (name + ".pdf")
        target.write_bytes(data)
        result.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), copy=str(target))
        try:
            from pypdf import PdfReader
            reader = PdfReader(target)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            text_path = ROOT / (name + ".txt")
            text_path.write_text(text, encoding="utf-8")
            result.update(pages=len(reader.pages), text_copy=str(text_path), status="retrieved_and_extracted")
        except Exception as error:
            result.update(status="retrieved_extraction_unavailable", extraction_error=repr(error))
    except Exception as error:
        result.update(status="retrieval_unavailable", error=repr(error))
    return result

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, SOURCES.items()))
    (ROOT / "download_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
