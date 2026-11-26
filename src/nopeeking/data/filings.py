"""List 8-K filings from the EDGAR submissions API and extract their text (8-K body + Exhibit 99.1)."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from nopeeking.data.sec import SecClient
from nopeeking.data.text import extract_8k_items, fix_c1, html_to_text, strip_boilerplate

log = logging.getLogger(__name__)

# Priority order: a filing with several items gets the first matching type.
EVENT_TYPES: list[tuple[str, frozenset[str]]] = [
    ("earnings", frozenset({"2.02"})),
    ("deal", frozenset({"1.01", "2.01"})),
    ("exec_change", frozenset({"5.02"})),
    ("other", frozenset({"8.01"})),
]
EVENT_PRIORITY = {name: i for i, (name, _) in enumerate(EVENT_TYPES)}
CACHE_CHARS = 40_000


def classify_event(items: str | None) -> str | None:
    codes = {c.strip() for c in (items or "").split(",") if c.strip()}
    for name, keys in EVENT_TYPES:
        if codes & keys:
            return name
    return None


def _submissions(client: SecClient, cik: int, start: str) -> tuple[dict, pd.DataFrame]:
    data = client.get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", max_age_days=7)
    frames = [pd.DataFrame(data["filings"]["recent"])]
    for extra in data["filings"].get("files", []):
        if extra.get("filingTo", "9999") < start:
            continue
        frames.append(
            pd.DataFrame(
                client.get_json(f"https://data.sec.gov/submissions/{extra['name']}", max_age_days=30)
            )
        )
    meta = {
        "name": data.get("name", ""),
        "tickers": data.get("tickers", []) or [],
        "former_names": [f["name"] for f in data.get("formerNames", []) or []],
    }
    return meta, pd.concat(frames, ignore_index=True)


def list_8k(client: SecClient, cik: int, start: str, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    meta, df = _submissions(client, cik, start)
    if df.empty or "form" not in df:
        return meta, pd.DataFrame()
    df = df[df["form"] == "8-K"].copy()
    if "items" not in df:
        df["items"] = ""
    df["acceptance_utc"] = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["acceptance_utc"])
    et_date = df["acceptance_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    df = df[(et_date >= pd.Timestamp(start)) & (et_date <= end)]
    out = pd.DataFrame(
        {
            "cik": cik,
            "accession": df["accessionNumber"].astype(str),
            "items": df["items"].fillna("").astype(str),
            "primary_document": df["primaryDocument"].fillna("").astype(str),
            "acceptance_utc": df["acceptance_utc"],
        }
    )
    return meta, out.reset_index(drop=True)


def filing_base_url(cik: int, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"


def parse_filing_index(html: bytes | str) -> list[dict[str, str]]:
    """Rows of the 'Document Format Files' table of an EDGAR ``-index.htm`` page."""
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="tableFile")
    if table is None:
        return []
    rows = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        link = tds[2].find("a")
        href = link.get("href", "") if link else ""
        if href.startswith("/ix?doc="):
            href = href[len("/ix?doc=") :]
        rows.append(
            {
                "description": tds[1].get_text(" ", strip=True),
                "href": href,
                "type": tds[3].get_text(" ", strip=True).upper(),
            }
        )
    return rows


def _pick(docs: list[dict[str, str]], exact: str, prefix: str) -> dict[str, str] | None:
    textual = [d for d in docs if d["href"].lower().endswith((".htm", ".html", ".txt"))]
    for d in textual:
        if d["type"] == exact:
            return d
    for d in textual:
        if d["type"].startswith(prefix):
            return d
    return None


def fetch_filing_text(client: SecClient, cik: int, accession: str, text_cache: Path) -> dict[str, str | bool]:
    """Return ``{"text_8k", "text_ex99", "has_ex99"}``; cached per accession as gzipped JSON."""
    path = text_cache / f"{accession}.json.gz"
    if path.exists():
        cached = json.loads(gzip.decompress(path.read_bytes()))
        return {**cached, "text_8k": fix_c1(cached["text_8k"]), "text_ex99": fix_c1(cached["text_ex99"])}
    base = filing_base_url(cik, accession)
    docs = parse_filing_index(client.get(f"{base}/{accession}-index.htm", use_cache=False))
    main = _pick(docs, "8-K", "8-K")
    ex99 = _pick(docs, "EX-99.1", "EX-99")
    text_8k = ""
    if main:
        text_8k = extract_8k_items(
            html_to_text(client.get("https://www.sec.gov" + main["href"], use_cache=False))
        )
    text_ex99 = ""
    if ex99:
        text_ex99 = html_to_text(client.get("https://www.sec.gov" + ex99["href"], use_cache=False))
    # cap what is cached: the dataset keeps at most `max_text_chars` (20k) of the combined text anyway
    result: dict[str, str | bool] = {
        "text_8k": text_8k[:CACHE_CHARS],
        "text_ex99": text_ex99[:CACHE_CHARS],
        "has_ex99": bool(ex99),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(result).encode()))
    tmp.replace(path)
    return result


def combine_text(text_8k: str, text_ex99: str, max_chars: int) -> str:
    text_8k, text_ex99 = strip_boilerplate(text_8k), strip_boilerplate(text_ex99)
    parts = []
    if text_8k:
        parts.append("[FORM 8-K]\n" + text_8k)
    if text_ex99:
        parts.append("[EXHIBIT 99.1]\n" + text_ex99)
    return "\n\n".join(parts)[:max_chars]
