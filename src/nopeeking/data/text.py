"""HTML -> text, 8-K item extraction, and anonymization (company names, tickers, dates, URLs)."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

from bs4 import BeautifulSoup, NavigableString, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_SPACES = re.compile(r"[ \t\r\f\v     ​ ]+")
_MANY_NL = re.compile(r"\n{3,}")
_DROP_CELLS = {"$", "%", ")", "(", "|"}


_C1 = re.compile("[-]")


def _c1_to_cp1252(m: re.Match) -> str:
    return bytes([ord(m.group(0))]).decode("cp1252", errors="ignore")


def fix_c1(text: str) -> str:
    """Repair Windows-1252 punctuation (smart quotes, dashes) that was decoded as Latin-1 control characters."""
    return _C1.sub(_c1_to_cp1252, text)


def normalize_ws(text: str) -> str:
    text = fix_c1(text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    text = "\n".join(lines)
    return _MANY_NL.sub("\n\n", text).strip()


_HEADER_LINE = re.compile(r"^(?:EX-\d+(?:\.\d+)?[A-Z]?|8-K|\d{1,3}|\S+\.(?:htm|html|txt))$", re.I)


def strip_sgml_header(text: str, max_lines: int = 8) -> str:
    """Drop EDGAR wrapper lines (TYPE / SEQUENCE / FILENAME) that precede the document body."""
    lines = text.split("\n")
    i = 0
    while i < min(max_lines, len(lines)) and (not lines[i].strip() or _HEADER_LINE.match(lines[i].strip())):
        i += 1
    return "\n".join(lines[i:]).strip()


def html_to_text(raw: bytes | str) -> str:
    """Visible text of an EDGAR HTML document; table rows become ``cell | cell`` lines."""
    if isinstance(raw, bytes):
        head = raw[:2000].lower()
        if b"<html" not in head and b"<div" not in head and b"<p" not in head and b"<table" not in head:
            return strip_sgml_header(normalize_ws(raw.decode("utf-8", errors="replace")))
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()
    for tag in soup.find_all(re.compile(r"^ix:header$", re.I)):
        tag.decompose()
    for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
        if not tag.decomposed:
            tag.decompose()
    for tr in soup.find_all("tr"):
        if tr.decomposed or tr.parent is None:
            continue
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in _DROP_CELLS]
        tr.replace_with(NavigableString("\n" + " | ".join(cells) + "\n" if cells else "\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for block in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        block.insert_after(NavigableString("\n"))
    return strip_sgml_header(normalize_ws(soup.get_text("")))


_ITEM_RE = re.compile(r"\bItem\s*(\d\.\d{2})", re.I)
_SIG_RE = re.compile(r"\n\s*SIGNATURES?\b", re.I)
_ITEM_901_RE = re.compile(r"\bItem\s*9\.01", re.I)


def extract_8k_items(text: str) -> str:
    """Body of a Form 8-K from its first ``Item x.xx`` heading to the exhibit list / signatures."""
    m = _ITEM_RE.search(text)
    if not m:
        return text
    body = text[m.start() :]
    cut = len(body)
    sig = _SIG_RE.search(body)
    if sig:
        cut = min(cut, sig.start())
    it901 = _ITEM_901_RE.search(body)
    if it901 and it901.start() > 0:
        cut = min(cut, it901.start())
    return body[:cut].strip()


_BOILERPLATE = re.compile(
    r"shall not be deemed|deemed (?:to be )?\W?filed\W? for (?:the )?purposes of Section 18|"
    r"incorporated by reference (?:into|in) any (?:filing|registration)|"
    r"forward[- ]looking statements?.{0,200}(?:risks?|uncertaint)|safe harbor|"
    r"undertakes? no (?:obligation|duty) to (?:update|revise)",
    re.I,
)


def strip_boilerplate(text: str, min_len: int = 120) -> str:
    """Drop long legal paragraphs (furnished-not-filed notices, forward-looking-statement disclaimers)."""
    keep = [line for line in text.split("\n") if len(line) < min_len or not _BOILERPLATE.search(line)]
    return _MANY_NL.sub("\n\n", "\n".join(keep)).strip()


# --------------------------------------------------------------------------------------------------
# Anonymization
# --------------------------------------------------------------------------------------------------

_SUFFIX_RE = re.compile(
    r"(?:,?\s+(?:inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|llc|lp|l\.p|n\.v|"
    r"s\.a|ag|se|holdings?|group|trust|bancorp|international|technologies|the|class\s+[a-c]|de|/[a-z]{2,3}/?)\.?)$",
    re.I,
)
_STATE_TAG_RE = re.compile(r"\s*/[A-Z]{2,3}/?\s*$")


def name_variants(names: Iterable[str], min_core_len: int = 4) -> list[str]:
    """Official names plus their 'core' (suffixes like Inc./Corp. stripped), longest first."""
    variants: set[str] = set()
    for raw in names:
        if not raw:
            continue
        name = _STATE_TAG_RE.sub("", raw).strip().rstrip(",")
        if not name:
            continue
        variants.add(name)
        core = name
        while True:
            stripped = _SUFFIX_RE.sub("", core).strip().rstrip(",")
            if stripped == core:
                break
            core = stripped
        core = re.sub(r"^the\s+", "", core, flags=re.I).strip()
        if len(core) >= min_core_len:
            variants.add(core)
    return sorted(variants, key=len, reverse=True)


_MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_YEAR = r"(?:19|20)\d{2}"
_DATE_PATTERNS = [
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+{_YEAR}\b"),
    re.compile(rf"\b\d{{1,2}}\s+{_MONTHS}\.?,?\s+{_YEAR}\b"),
    re.compile(rf"\b{_MONTHS}\.?\s+{_YEAR}\b"),
    re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/(?:19|20)?\d{2}\b"),
    re.compile(rf"\b{_YEAR}-\d{{2}}-\d{{2}}\b"),
    re.compile(r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"),
]
_SHORT_YEAR_RE = re.compile(r"\b(FY|Q[1-4]|H[12])\s?'?(\d{2})\b")
_APOS_YEAR_RE = re.compile(r"(?<=\s)'\d{2}\b")
_FULL_YEAR_RE = re.compile(rf"\b{_YEAR}\b")
_EXCHANGE_TICKER_RE = re.compile(
    r"\b(NYSE(?:\s+American)?|NASDAQ|Nasdaq(?:GS|GM)?|Cboe|CBOE|NYSE\s+Arca)\s*:\s*\"?[A-Z][A-Z0-9.\-]{0,6}\"?"
)
_URL_RE = re.compile(
    r"\b(?:https?://)?(?:www\.)?[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9\-]+)*\.(?:com|net|org|io|co)\b\S*", re.I
)
_EMAIL_RE = re.compile(r"\b[\w.+\-]+@[\w\-]+\.[\w.\-]+\b")
_FILENAME_RE = re.compile(r"\b[\w\-]+\.(?:htm|html|txt|xml|jpg|png)\b", re.I)


def anonymize(text: str, names: Iterable[str], tickers: Iterable[str]) -> str:
    """Mask company names, tickers, dates, years, URLs and e-mails.

    This is deliberately simple and is *expected* to leak (executive names, products, numbers):
    measuring how much identity survives masking is part of the study.
    """
    out = _EMAIL_RE.sub("[EMAIL]", text)
    out = _FILENAME_RE.sub("[FILE]", out)
    out = _URL_RE.sub("[URL]", out)
    out = _EXCHANGE_TICKER_RE.sub(r"\1: [TICKER]", out)
    for variant in name_variants(names):
        out = re.sub(rf"(?<!\w){re.escape(variant)}(?:'s|’s)?(?!\w)", "[COMPANY]", out, flags=re.I)
    for ticker in sorted({t for t in tickers if t}, key=len, reverse=True):
        if len(ticker) >= 3:
            out = re.sub(rf"(?<![\w\[]){re.escape(ticker)}(?![\w\]])", "[TICKER]", out)
    for pat in _DATE_PATTERNS:
        out = pat.sub("[DATE]", out)
    out = _SHORT_YEAR_RE.sub(r"\1 [YEAR]", out)
    out = _FULL_YEAR_RE.sub("[YEAR]", out)
    out = _APOS_YEAR_RE.sub("[YEAR]", out)
    return out
