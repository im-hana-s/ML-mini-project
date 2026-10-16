# Data instructions

Raw data is excluded by default. With permission to redistribute, this repository includes these source files:

- `raw/gvp_eruption_list.xlsx`
- `raw/wovodat_data_type_statistics.csv.csv`
- `raw/wovodat_volcano_statistics.csv.csv`

Do not add temporary editor files or other datasets without checking their redistribution terms. Cite the data sources in the project report and document the study region, download dates and filters.

## `raw/earthquakes.csv` required columns
- `time`: timestamp parseable by pandas; use ISO 8601/UTC, e.g. `2018-05-01T12:30:00Z`
- `latitude`: decimal degrees
- `longitude`: decimal degrees
- `depth`: depth in km
- `magnitude` (or USGS `mag`): event magnitude

## `raw/eruptions.csv` required columns
- `start_time`: eruption start timestamp
- `end_time`: eruption end timestamp

Use an earthquake catalog for the study volcano and a period that overlaps the eruption intervals. A recent global catalog is not a substitute: earthquakes outside the study area or outside the eruption date range cannot support this project's labels. To download the project's Kīlauea catalog for 1950–2018 from the USGS, run `python src/download_earthquakes.py`. It requests events within 0.5 degrees of Kīlauea with magnitude 3 or greater and replaces `raw/earthquakes.csv` only after validating the download. This broader window includes both repose and erupting intervals; the 2010–2018 window alone falls entirely within the 1983–2018 eruption.

Each row represents one documented eruption interval. This starter script requires an end time for every interval. For ongoing/open intervals, define and document a defensible study cutoff before labeling; do not invent an end date.

The original paper reports a 3,764-earthquake dataset, but a newly downloaded catalog may differ. Record actual data source, download date, time range, filters and row counts in the report. Never claim an exact reproduction unless the data and filtering match the paper.
