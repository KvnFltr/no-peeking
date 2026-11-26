# Pre-registration

Written on 2026-11-25, **before any model was evaluated on a test split**. Changes after that date are listed
in the changelog at the bottom with the reason. The post-cutoff test set is scored **once per method**; everything
that is tuned (hyper-parameters, prompts, thresholds, tercile cut-offs) is tuned on `val` (2024) only.

## Data and splits (fixed)

| split | filing dates (ET) | role |
|---|---|---|
| train | 2018-01-01 to 2022-12-31 | fitting / post-training |
| test_pre | 2023-01-01 to 2023-12-31 | pre-cutoff test, **only** for the memorization gap (RQ1) |
| val | 2024-01-01 to 2024-12-31 | all tuning and model selection |
| (embargo) | 2025-01-01 to 2025-06-30 | unused, so that 2025 knowledge of recent models cannot leak |
| test_post | 2025-07-01 onwards | headline out-of-sample test, grows every month |

Target: sign of the 3-day market-adjusted return (stock minus SPY) over sessions [t, t+2], t = first session whose
16:00 ET close is after the EDGAR acceptance time. Secondary: 1-day sign and the 3-way tercile (cut-offs from train).

## RQ1 - look-ahead bias (zero-shot models)

* Primary estimate: **DiD of AUC** = (AUC pre,orig - AUC pre,anon) - (AUC post,orig - AUC post,anon), 95% bootstrap CI
  (2,000 resamples, filings resampled within split, original/anonymized paired).
* Control: the same DiD for TF-IDF + logistic regression trained on 2018-2022 (cannot remember 2023), which
  measures how much anonymization removes legitimate information.
* Reading: a DiD whose CI excludes 0 and exceeds the TF-IDF control is evidence of recall. A null result is reported as
  a null result, with its CI.
* Models: Qwen2.5-0.5B/1.5B/7B-Instruct (cut-off before 2025-07) plus any model submitted through `nopeeking probe`.

## RQ2 - hindsight distillation

* Compare on `test_post`: SFT on hindsight rationales vs. SFT on control rationales (written without the outcome)
  vs. direct-label SFT vs. zero-shot student.
* Primary metric: **AUC**. Co-primary: **Brier score and ECE** (calibration). The hypothesis "hindsight teaches
  overconfident stories" predicts a worse Brier/ECE for hindsight-SFT than for control-SFT at similar AUC.
* Differences: paired bootstrap over test filings, 95% CI; 3 seeds for the main runs (mean and spread reported).

## RQ3 - RL

* GRPO variants from the hindsight-SFT adapter: (a) outcome reward (Brier-based, proper), (b) + grounding reward,
  (c) + format reward; DPO on (correct vs wrong) rationale pairs.
* Primary: AUC and Brier on `test_post`; faithfulness: fact precision of cited numbers and leak rate.
* Budget: identical number of optimizer steps and effective batch size per method, fixed after the pilot and
  recorded here before the main runs. Wall-clock GPU time is reported for every run.

## Finance

Long-short, market-hedged, top vs. bottom tercile of the score (cut-offs from `val` scores), entered at the first close
after the filing (close of day t) and held over sessions t+1 and t+2, 5 bps per side. Annualized Sharpe with a circular block bootstrap CI (block 5 days). Reported for every method, never used for
selection.

## Changelog

* 2026-11-25: initial version.
* 2026-11-25, pilot (pipeline check, 493 filings): legal boilerplate (furnished-not-filed notices, forward-looking
  statement disclaimers) is stripped from filing text before any model saw it. After the first pilot report, the
  majority baseline was changed to output the train base rate instead of a hard 0/1 probability (accuracy unchanged,
  Brier/ECE now meaningful). The pilot's seed-0 train sample has an up-rate of 0.407 versus 0.513 for its full
  candidate pool (other seeds: 0.51-0.57); the seed was kept. Pilot test filings were looked at by the baselines only;
  the main results use the full dataset and are scored once per method.
* 2026-11-26, v1 (19,783 filings): the first v1 portfolio run entered positions at the close of t-1, before the filing
  was public, which is a look-ahead error in the simulator (it reported a TF-IDF Sharpe ratio of 2.12). Positions now
  enter at the close of t and earn t+1..t+2 only; this was fixed before any language model was evaluated, and the
  prediction target itself is unchanged.
