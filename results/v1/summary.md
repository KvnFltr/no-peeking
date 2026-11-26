### Leaderboard: post-cutoff test

Ranked by AUC on filings published after every model's release (unseen by construction). DiD > 0 means the model does better on pre-cutoff filings only when it can see who and when: recall.

| # | model | released | params | AUC post-cutoff [95% CI] | Brier | DiD (memorization) [95% CI] |
|---:|---|---|---:|---|---|---|
| 1 | TF-IDF + LR (trained on 2018-22) |  |  | 0.537 [0.522, 0.551] | 0.250 | -0.002 [-0.007, 0.003] |
| 2 | Qwen/Qwen2.5-0.5B-Instruct | 2024-09 | 0.5B | 0.517 [0.482, 0.552] | 0.411 | -0.033 [-0.081, 0.011] |

Dataset `v1`: 19783 filings, 2018-01-02 to 2026-11-23. Split sizes: train 7948, test_pre 3979, val 1992, test_post 5864.

**Filter log**

| step | before | after | note |
|---|---:|---:|---|
| universe firms (unique CIK) |  | 899 | sp500+sp400 |
| 8-K filings in date range |  | 95719 | 2018-01-01..2026-11-25 |
| event type in scope (items 2.02/1.01/2.01/5.02/8.01) | 95719 | 76113 |  |
| inside a split window | 76113 | 71764 | drops the embargo gap between val and test_post |
| prices available for full label window | 71764 | 71325 | needs close(t-1)..close(t+2); recent filings wait for their window |
| one filing per firm and event day | 71325 | 69469 | keeps highest-priority event type |
| quota sample per split | 69469 | 20000 | {"train": 8000, "test_pre": 4000, "val": 2000, "test_post": 6000} |
| filing text fetched | 20000 | 19867 | 133 fetch failures |
| text length >= 300 chars | 19867 | 19783 |  |
| dedup identical texts | 19783 | 19783 |  |

**Share of filings followed by a 3-day outperformance (y_up = 1)**: test_post 0.481, test_pre 0.499, train 0.511, val 0.468

**post-cutoff test (2025-07+)**: metric [95% bootstrap CI]

| method | n | AUC | balanced acc. | acc. | Brier | ECE | rank IC |
|---|---:|---|---|---|---|---:|---|
| majority | 5864 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.481 [0.469, 0.494] | 0.251 [0.250, 0.251] | 0.029 | – [–, –] |
| event_type_prior | 5864 | 0.524 [0.510, 0.538] | 0.524 [0.513, 0.533] | 0.513 [0.500, 0.525] | 0.250 [0.250, 0.251] | 0.030 | 0.042 [0.021, 0.063] |
| tfidf_lr | 5864 | 0.537 [0.522, 0.551] | 0.520 [0.508, 0.531] | 0.511 [0.498, 0.524] | 0.250 [0.249, 0.250] | 0.031 | 0.068 [0.043, 0.092] |
| tfidf_lr_anon | 5864 | 0.536 [0.521, 0.550] | 0.523 [0.511, 0.534] | 0.515 [0.501, 0.527] | 0.250 [0.249, 0.250] | 0.031 | 0.067 [0.041, 0.091] |
| Qwen/Qwen2.5-0.5B-Instruct (zero-shot) | 1000 | 0.517 [0.482, 0.552] | 0.500 [0.500, 0.500] | 0.526 [0.495, 0.558] | 0.411 [0.384, 0.437] | 0.402 | 0.032 [-0.029, 0.096] |

**pre-cutoff test (2023)**: metric [95% bootstrap CI]

| method | n | AUC | balanced acc. | acc. | Brier | ECE | rank IC |
|---|---:|---|---|---|---|---:|---|
| majority | 3979 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.499 [0.484, 0.515] | 0.250 [0.250, 0.250] | 0.011 | – [–, –] |
| event_type_prior | 3979 | 0.511 [0.495, 0.529] | 0.509 [0.497, 0.522] | 0.509 [0.494, 0.524] | 0.250 [0.250, 0.250] | 0.012 | 0.024 [-0.001, 0.052] |
| tfidf_lr | 3979 | 0.530 [0.512, 0.548] | 0.514 [0.500, 0.527] | 0.513 [0.498, 0.529] | 0.249 [0.249, 0.250] | 0.012 | 0.062 [0.031, 0.092] |
| tfidf_lr_anon | 3979 | 0.532 [0.514, 0.549] | 0.512 [0.498, 0.525] | 0.511 [0.496, 0.527] | 0.249 [0.249, 0.250] | 0.011 | 0.065 [0.034, 0.095] |
| Qwen/Qwen2.5-0.5B-Instruct (zero-shot) | 1000 | 0.507 [0.470, 0.543] | 0.500 [0.500, 0.500] | 0.502 [0.471, 0.532] | 0.430 [0.404, 0.456] | 0.424 | -0.004 [-0.068, 0.056] |

**val (2024)**: metric [95% bootstrap CI]

| method | n | AUC | balanced acc. | acc. | Brier | ECE | rank IC |
|---|---:|---|---|---|---|---:|---|
| majority | 1992 | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | 0.468 [0.447, 0.490] | 0.251 [0.250, 0.251] | 0.042 | – [–, –] |
| event_type_prior | 1992 | 0.502 [0.478, 0.527] | 0.513 [0.495, 0.532] | 0.496 [0.474, 0.518] | 0.251 [0.250, 0.251] | 0.042 | 0.011 [-0.026, 0.047] |
| tfidf_lr | 1992 | 0.550 [0.524, 0.575] | 0.523 [0.504, 0.542] | 0.508 [0.485, 0.530] | 0.250 [0.249, 0.251] | 0.045 | 0.073 [0.029, 0.117] |
| tfidf_lr_anon | 1992 | 0.550 [0.524, 0.575] | 0.519 [0.500, 0.538] | 0.504 [0.482, 0.526] | 0.250 [0.249, 0.251] | 0.044 | 0.073 [0.029, 0.116] |
| Qwen/Qwen2.5-0.5B-Instruct (zero-shot) | 1000 | 0.492 [0.456, 0.529] | 0.499 [0.497, 0.500] | 0.526 [0.495, 0.558] | 0.412 [0.385, 0.439] | 0.402 | -0.030 [-0.093, 0.030] |

**Look-ahead probe (AUC)**: DiD = (pre orig - pre anon) - (post orig - post anon); > 0 suggests recall of pre-cutoff outcomes.

| model | pre orig | pre anon | post orig | post anon | pre - post (orig) | DiD |
|---|---|---|---|---|---|---|
| TF-IDF + LR (control, trained <= 2022) | 0.530 [0.513, 0.549] | 0.532 [0.514, 0.550] | 0.537 [0.522, 0.551] | 0.536 [0.521, 0.550] | -0.006 [-0.029, 0.018] | -0.002 [-0.007, 0.003] |
| Qwen/Qwen2.5-0.5B-Instruct | 0.507 [0.472, 0.541] | 0.515 [0.480, 0.550] | 0.517 [0.482, 0.552] | 0.492 [0.456, 0.526] | -0.010 [-0.060, 0.044] | -0.033 [-0.081, 0.011] |

**Long-short on post-cutoff test** (top vs bottom tercile of score, cut-offs from val, 5 bps per side, market-hedged)

| method | positions | active days | ann. return | ann. vol | Sharpe [95% CI] |
|---|---:|---:|---:|---:|---|
| event_type_prior | 5864 | 353 | -0.230 | 0.164 | -1.40 [-3.15, 0.24] |
| tfidf_lr | 4032 | 353 | 0.139 | 0.233 | 0.60 [-1.17, 2.12] |
| tfidf_lr_anon | 4064 | 353 | 0.194 | 0.225 | 0.86 [-0.88, 2.36] |
| Qwen/Qwen2.5-0.5B-Instruct (zero-shot) | 658 | 336 | -0.379 | 0.355 | -1.07 [-2.74, 0.65] |
