"""Download candidate articles and pull out the main text.

Article HTML is untrusted input: it is parsed by trafilatura into plain text
and only ever handed to the extraction model as quoted data.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd
import trafilatura

from ... import config

CACHE = config.RAW / "articles"

ASSET_RE = re.compile(
    r"wind|turbine|solar|photovolt|panel|energy|power plant|"
    r"éolien|solaire|eólic|solar|windkraft|windpark|風力|太陽光|광",
    re.I,
)
DAMAGE_RE = re.compile(
    r"fire|blaze|burn|explo|collaps|topple|fell|hail|tornado|storm|damage|destroy|lightning|"
    r"smoke|evacuat|failure|broke|snapped|incendi|feu|brand|sturm|hagel|火|爆|화재|폭발",
    re.I,
)


def _key(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:20]


def prefilter(articles: pd.DataFrame) -> pd.DataFrame:
    """Cheap title screen before spending an LLM call on an article."""
    t = articles["title"].fillna("")
    keep = t.str.contains(ASSET_RE) & t.str.contains(DAMAGE_RE)
    return articles[keep]


def fetch_one(url: str) -> dict:
    path = CACHE / f"{_key(url)}.json"
    if path.exists():
        return json.loads(path.read_text())
    rec = {"url": url, "ok": False, "text": "", "error": None}
    try:
        r = httpx.get(
            url,
            timeout=25,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; rrisk/0.1; research)"},
        )
        r.raise_for_status()
        if "html" not in r.headers.get("content-type", "html"):
            raise ValueError(f"non-HTML content: {r.headers.get('content-type')}")
        text = trafilatura.extract(r.text, include_comments=False, include_tables=False) or ""
        rec.update(ok=bool(text.strip()), text=text[:20000])
    except Exception as e:  # network errors, 4xx/5xx, parser failures
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec))
    return rec


def fetch(articles: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(fetch_one, articles["url"].tolist()))
    texts = pd.DataFrame(recs)[["url", "ok", "text", "error"]]
    out = articles.merge(texts, on="url", how="left")
    print(f"[fetch] {out['ok'].sum():,}/{len(out):,} articles with usable text")
    return out
