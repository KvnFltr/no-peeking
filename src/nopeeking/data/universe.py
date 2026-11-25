"""Firm universe: current S&P 500 / S&P 400 constituents mapped to SEC CIKs.

Using *current* constituents introduces survivorship bias for past years; this is documented as a
limitation. Snapshots are written to ``data/universe/`` so every run is reproducible.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from nopeeking.data.sec import SecClient, user_agent

log = logging.getLogger(__name__)

WIKIPEDIA = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
}
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"


def _fetch_wikipedia_table(index: str) -> pd.DataFrame:
    resp = requests.get(WIKIPEDIA[index], headers={"User-Agent": user_agent()}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        sym = cols.get("symbol") or cols.get("ticker symbol") or cols.get("ticker")
        name = cols.get("security") or cols.get("company")
        if sym is not None and name is not None and len(table) > 100:
            out = pd.DataFrame(
                {"ticker": table[sym].astype(str).str.strip(), "name": table[name].astype(str)}
            )
            out["index"] = index
            return out
    raise RuntimeError(f"No constituents table found on {WIKIPEDIA[index]}")


def _sec_ticker_map(client: SecClient) -> dict[str, int]:
    data = client.get_json(SEC_TICKERS, max_age_days=30)
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def load_universe(indexes: list[str], client: SecClient, universe_dir: str | Path) -> pd.DataFrame:
    """Return one row per firm (deduplicated on CIK): ticker, name, cik, index."""
    universe_dir = Path(universe_dir)
    universe_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in indexes:
        snaps = sorted(universe_dir.glob(f"{index}_*.csv"))
        if snaps:
            df = pd.read_csv(snaps[-1], dtype={"ticker": str, "cik": "Int64"})
            log.info("universe %s: using snapshot %s", index, snaps[-1].name)
        else:
            df = _fetch_wikipedia_table(index)
            mapping = _sec_ticker_map(client)
            df["cik"] = (
                df["ticker"].map(lambda t, m=mapping: m.get(t.upper().replace(".", "-"))).astype("Int64")
            )
            missing = df["cik"].isna().sum()
            if missing:
                log.warning(
                    "universe %s: %d tickers without CIK dropped: %s",
                    index,
                    missing,
                    df.loc[df["cik"].isna(), "ticker"].tolist(),
                )
            df = df.dropna(subset=["cik"])
            snap = universe_dir / f"{index}_{pd.Timestamp.today():%Y-%m-%d}.csv"
            df.to_csv(snap, index=False)
            log.info("universe %s: wrote snapshot %s (%d firms)", index, snap.name, len(df))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["cik"] = out["cik"].astype(int)
    return out.drop_duplicates(subset="cik", keep="first").reset_index(drop=True)
