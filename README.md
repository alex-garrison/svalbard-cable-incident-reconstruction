# Svalbard Cable Incident Reconstruction

This repository contains the reproducible analysis for my reconstruction of the 7 January 2022 failure of Svalbard's cable A. It tests whether the *Melkart-5*'s trawl gear plausibly crossed the cable during the reported failure minute.

The two-dimensional sensitivity model estimates the horizontal paths of the port door, starboard door and centre clump. It produces cable crossings for part of the tested parameter range, but does not establish damage, probability, or intent.

## Reproduce

```bash
uv sync --locked
uv run svalbard-cable reproduce
```

To include Global Fishing Watch metrics, set `GFW_API_ACCESS_TOKEN` in the environment or in the repository's `.env` file.

This acquires and validates the article's AIS, cable, bathymetry, EFS, ERS and fishing-activity inputs, then writes the provenance, results and publication metrics to `db/svalbard_cable_incident_reconstruction.duckdb`. Configuration variables are in `config/config.toml`.

The command writes the results to `outputs/reproduction_results.json`. Use `reproduce --output PATH` to select another file.

## Playback export

After reproduction, run `uv run svalbard-cable export-playback` to write the website playback data to `outputs/playback_data.json`. Use `export-playback --output PATH` to select another file.
