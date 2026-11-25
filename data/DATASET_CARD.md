---
license: mit
language: [en]
task_categories: [text-classification]
tags: [finance, sec, 8-k, look-ahead-bias, llm-evaluation, benchmark]
pretty_name: NoPeek-8K
---

# NoPeek-8K

SEC Form 8-K filings (with their Exhibit 99.1 press releases) from S&P 500 and S&P 400 companies, each paired with
the stock's market-adjusted reaction over the next three trading sessions. The post-cutoff test split only contains
filings published after the training cutoff of the models studied, and grows every month.

Built with [`nopeeking`](https://github.com/KvnFltr/no-peeking): `nopeeking data build --config configs/full.yaml`.

## Fields

| field | description |
|---|---|
| `id`, `accession`, `cik`, `ticker`, `company` | EDGAR identifiers |
| `items`, `event_type` | 8-K item codes; earnings (2.02) > deal (1.01/2.01) > exec_change (5.02) > other (8.01) |
| `acceptance_utc`, `filing_date`, `event_day`, `window_end` | EDGAR acceptance time; first session closing after it; end of the 3-day window |
| `split` | train (2018-22), test_pre (2023), val (2024), test_post (2025-07 onwards) |
| `text`, `text_8k`, `text_ex99`, `has_ex99` | extracted filing text (8-K items + press release), truncated to 20,000 characters |
| `ar_1d`, `ar_3d`, `ar_3d_w`, `ar_1d_w` | stock return minus SPY return (raw and winsorized at train 1%/99%) |
| `y_up`, `y_up_1d`, `y_tercile` | targets: 3-day sign, 1-day sign, tercile with train cut-offs |
| `aliases`, `tickers_all` | names and tickers used by the anonymizer |

## Conventions and hygiene

* Event day `t`: first trading session whose 16:00 ET close is after the acceptance timestamp. 3-day window = sessions
  [t, t+2], returns from close(t-1).
* One filing per firm and event day (highest-priority event type kept), exact-text duplicates removed, filings without
  prices for the whole window dropped. Every filter is logged with counts in `filter_log.json`.
* Raw prices are not included. Labels are derived from daily adjusted closes (Yahoo Finance via `yfinance`); rebuild
  them with the scripts.

## Known limitations

Survivorship bias (current index members), anonymization that does not mask people or products, close-to-close
returns that include intraday pre-filing moves for filings accepted during market hours. Not financial advice.
