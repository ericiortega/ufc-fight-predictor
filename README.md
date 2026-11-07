# UFC Fight Predictor

An end-to-end data science project that predicts UFC fight outcomes and win
probabilities, then re-ingests results and retrains after each card.

**This is a research and portfolio project, not a betting tool.** Outputs are
model predictions and research findings, never picks or betting advice.

> **Work in progress.** Built in phases: ingestion, point-in-time features and
> Elo, walk-forward evaluation, and pre-event prediction logging are done; the
> retraining pipeline, app, enrichment study and full write-up are in progress.
>
> Pre-event predictions are appended to `predictions/history.jsonl`, a
> hash-chained log. Each line's hash covers the previous line's hash, and the
> git push time records that predictions existed before each event.

## Data sources

- **Fight and fighter data:** Data from Greco1899/scrape_ufc_stats (GPL-3.0),
  originally scraped from ufcstats.com.
  <https://github.com/Greco1899/scrape_ufc_stats>
  Each snapshot is downloaded at a pinned upstream commit. Its manifest
  (`data/manifests/greco/`, committed) records the commit SHA and file checksums,
  so any past snapshot can be re-downloaded byte for byte.
- **Upcoming events and fight cards:** Wikipedia (CC BY-SA 4.0), via the
  MediaWiki API. Each card records the exact page revision it was read from.

ufcstats.com is not scraped directly: it serves a JavaScript bot challenge to
automated clients, and this project does not bypass access controls.

### Backup source (not implemented)

If the upstream repository stops updating, the staleness check fails the
pipeline loudly. A fallback candidate is a Kaggle UFC dataset such as the
*Ultimate UFC Dataset* (mdabbert). Its license would need to be checked before use.

## Fighter identity matching

Upstream fight rows identify fighters by name only, and Wikipedia spells names
differently again. `ufc_model.names` resolves names conservatively:

1. Curated alias (`config/name_aliases.yaml`, human-reviewed, optionally scoped by weight class).
2. Exact match on the profile name or ufcstats' alternate name.
3. Normalized match (accents, case, punctuation, Jr./III suffixes), then spaces removed.
4. Several fighters with the same name: tie-break on plausible age, then weight class (with a margin).
5. Otherwise **unresolved**, with fuzzy suggestions logged to `name_match_log`. Fuzzy matches are never auto-accepted.

An upcoming bout is predicted only if both fighters resolve confidently.
Otherwise it is marked `predictable = false` with a reason.

## How the model works

### 1. Data

Six upstream CSVs (events, fights, results, round-by-round stats, fighter
details, and height/reach/stance/date of birth) are loaded into DuckDB
(`data/ufc.duckdb`). Every UFC fight from 1994 on is used to build fighter
history and Elo. Training and evaluation use only fights from 2005 on that had a
winner and where both fighters resolved: 8,382 fights up to the 2026-10-31 data
cutoff. Draws and no-contests feed fighter history, but they are not training
targets.

### 2. Point-in-time features

Features are built in one pass over event dates (`ufc_model.features.build`).
For each date, every fighter on the card is first described using only fights
from earlier dates. Only after that are that date's results added. Nothing from
the fight day or later can reach a feature, and leakage tests check this.

For each fighter:

| Group | Features |
|:--|:--|
| Record | fights, wins, losses, win rate, current streak, recent win rate (last 3), debut flag, days since last fight |
| Finishing | share of wins by KO/submission, rate of being finished |
| Output and defense (career and last 3 fights) | significant strikes landed/absorbed per minute, striking accuracy and defense, takedowns per 15 min, takedown accuracy and defense, submission attempts, control share, knockdowns scored and absorbed |
| Rating | Elo (K = 80, tuned on 2005-2012; a finish bonus did not help) |
| Profile | height, reach, age at fight time, stance |

- **Shrinkage.** Rates are shrunk toward the league average as it stood at the
  time, so a fighter with one 30-second fight does not get an extreme striking
  rate.
- **Missing profiles.** Missing height, reach, stance or age is filled from the
  same weight class's fighters seen before that date. The model never sees a
  "missing" flag. Profiles are scraped today, and missing values tend to belong
  to short careers, so a flag would leak the future.
- **Both fighters, both directions.** Each bout becomes two mirrored rows
  (A minus B and B minus A) with the label flipped, plus context: weight class,
  title fight, women's bout, five rounds, and stance matchup. That gives 69
  model inputs. The final probability averages the two directions, so
  p(A) + p(B) = 1 and corner order cannot matter.

### 3. Evaluation

Models were compared walk-forward (`ufc_model.models.evaluate`). For each
year from 2013 to 2026, the model trains on all earlier years and predicts that
year. Hyperparameters (Optuna) and calibration are tuned on the last two years
of each training window, never on the test year. Pooled over 6,684 test fights:

| Model | Accuracy | Log loss | Brier |
|:--|--:|--:|--:|
| Logistic regression | 61.9% | 0.653 | 0.231 |
| LightGBM (5-seed average) | 61.1% | 0.655 | 0.232 |
| Elo alone | 55.9% | 0.682 | 0.244 |
| Pick the younger fighter | 57.8% | - | - |
| Pick the higher win rate | 57.2% | - | - |

- Calibration (Platt, isotonic, temperature) did not improve on the raw model,
  so the production model is uncalibrated.
- Dropping height, reach and age costs LightGBM about 1.7 accuracy points.
- Starting training in 2001 instead of 2005 made no difference.
- Age and Elo are the strongest drivers by global SHAP, followed by strikes
  absorbed per minute and takedown rate.

Per-year tables, subsets and reliability plots are in
[`reports/phase3/summary.md`](reports/phase3/summary.md) and
`reports/phase3/figures/`.

### 4. Production model and predictions

- **Model.** Standardized features and L2 logistic regression
  (`ufc_model.models.production`), refit on all eligible fights up to the data
  cutoff.
- **Recent-card check.** Each retrain is also checked on the most recent year
  of cards, using a model trained only on fights before that year. For
  `lr-20261031-9668ddd9`: 64.0% accuracy and log loss 0.640, against Elo's
  55.5% and 0.687. A temperature fitted on that year stays near 1, which means
  there is no calibration drift.
- **Upcoming cards.** Cards come from Wikipedia. Both fighters are described
  from the current state, scored, and logged with the top five drivers. Drivers
  are exact linear SHAP values in log-odds, and a positive value favors the
  first-named fighter. Elo's probability is logged alongside as the baseline.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m ufc_model.ingest --limit-events 2   # trial run on the 2 latest events
.venv/bin/python -m ufc_model.ingest                    # full history
.venv/bin/pytest
```

## License

GPL-3.0 (see `LICENSE`), matching the upstream dataset's license.
