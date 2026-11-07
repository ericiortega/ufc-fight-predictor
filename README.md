# UFC Fight Predictor

An end-to-end data science project that predicts UFC fight outcomes and win
probabilities, then re-ingests results and retrains after each card.

**This is a research and portfolio project, not a betting tool.** Outputs are
model predictions and research findings, never picks or betting advice.

> Status: Phase 1 (data ingestion). The full README (methodology, leakage
> prevention, results vs. baselines, ablation, limitations) arrives in Phase 7.

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

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m ufc_model.ingest --limit-events 2   # trial run on the 2 latest events
.venv/bin/python -m ufc_model.ingest                    # full history
.venv/bin/pytest
```

## License

GPL-3.0 (see `LICENSE`), matching the upstream dataset's license.
