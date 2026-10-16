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
- `magnitude`: event magnitude

## `raw/eruptions.csv` required columns
- `start_time`: eruption start timestamp
- `end_time`: eruption end timestamp

Each row represents one documented eruption interval. This starter script requires an end time for every interval. For ongoing/open intervals, define and document a defensible study cutoff before labeling; do not invent an end date.

The original paper reports a 3,764-earthquake dataset, but a newly downloaded catalog may differ. Record actual data source, download date, time range, filters and row counts in the report. Never claim an exact reproduction unless the data and filtering match the paper.
