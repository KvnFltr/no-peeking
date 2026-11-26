# Launch kit

Everything here is a draft for the author to post from their own accounts. Replace every `{…}` with numbers from
`results/v1/summary.md` (never round them in your favour) and only post once the main runs are in the README.

## 1. arXiv

* Build: `cd paper && latexmk -pdf main.tex` (or download the `Paper` workflow artifact), then
  `python scripts/make_arxiv.py --results results/v1 --bbl paper/build/main.bbl` creates `outputs/arxiv.tar.gz`.
* Categories: primary **cs.CL**, cross-list **q-fin.CP** and **cs.LG**. A first-time submitter may be asked for an
  endorsement (arxiv.org/auth/endorse), depending on the category and account: a supervisor or any author with recent
  cs.CL papers can endorse.
* Licence: CC BY 4.0. Comments field: "Code and data: https://github.com/KvnFltr/no-peeking".
* After the arXiv ID exists: add it to `CITATION.cff`, the README badge and the Hugging Face dataset card.

## 2. Venues (after arXiv)

* NeurIPS **Datasets & Benchmarks** track (style option `dandb`), or the ICLR/ICML workshops on financial AI and on
  data contamination; finance venues: the ACM ICAIF conference.

## 3. Hugging Face

* `hf auth login`, then `python scripts/publish_hf.py --config configs/v1.yaml --repo <user>/NoPeek-8K`.
* Link the dataset to the paper page (huggingface.co/papers/<arxiv-id>) so it appears under "Datasets citing this paper".

## 4. Posts (drafts)

**X / Bluesky thread**

1. Do LLMs forecast how stocks react to company filings, or do they just remember what happened? We built a
   benchmark that cannot be memorized: SEC 8-K filings published after every model's release, refreshed monthly. 🧵
2. Probe: same filings, original vs. anonymized, before vs. after the cutoff. Memorization shows up as a
   difference-in-differences: {headline DiD with CI, model}.
3. On the unseen post-cutoff test, the best open model reaches AUC {x} [{lo}, {hi}] vs. {y} for a TF-IDF baseline.
4. Teaching a 0.5B model with "hindsight" rationales: {calibration result}.
5. One command tests any HF model: `nopeeking probe <model>`. Leaderboard open to PRs. Paper + code + data: {links}

**LinkedIn (French)**

> Les LLM savent-ils prédire la réaction des marchés aux annonces des entreprises, ou s'en souviennent-ils ?
> J'ai construit No Peeking, un benchmark de dépôts SEC 8-K qui se met à jour chaque mois, pour qu'aucun modèle n'ait
> pu le mémoriser. Résultat principal : {…}. Le code, les données et un leaderboard ouvert sont en ligne : {lien}.

**Reddit r/MachineLearning ([R] flair)** — title: "[R] No Peeking: a self-refreshing benchmark for look-ahead bias in
LLM market forecasts (code, data, leaderboard)". Body: 5 lines of findings with CIs, the figure, links, and an explicit
"limitations" paragraph (readers there check for overclaiming).

**Hacker News** — "Show HN: Test whether your LLM remembers the stock market instead of predicting it".

## 5. Make it useful to others (what actually brings stars and forks)

* Keep the monthly refresh green: a benchmark that stays fresh is the reason to come back.
* Answer leaderboard PRs fast; tag `good first issue`s (new models, new event types, other markets).
* Ask authors of related papers (look-ahead bias, point-in-time LLMs) whether they want their model on the leaderboard.
