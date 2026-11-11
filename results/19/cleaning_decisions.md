# Audit and cleaning decisions (Group 19)

## Dataset
- File: data/raw/combined_gee_pm25.csv (teacher-provided merge). All three raw CSVs are untouched.
- Source: https://www.kaggle.com/datasets/sarahameziane/air-quality-mediterranean-basin-20162026
- License: the dataset description states that the raw data come from OpenAQ (CC BY 4.0), Copernicus open-access and NASA open data, and asks users to check the individual source licenses before commercial use.
- Units (per the dataset description): pm25 ug/m3; temperature degC; pressure hPa; wind m/s; NO2, CO, O3 mol/m2; AOD dimensionless.
- 41,718 rows, 15 columns; 2016-11-25 to 2026-04-06; 94 sensor names, 98 GEE ids; lat 36.13 to 43.53, lon -5.66 to 9.14; PM2.5 in ug/m3.

## Decisions
1. The combined file is used as provided; the merge was not rebuilt. Known limits: incomplete daily coverage (median sensor day-coverage 50%); diagnostics found 5,849 potentially matchable observations absent from the combined file (not shown to be wrongly excluded).
2. Station = sensor_name (column station_id after cleaning). The GEE id is kept only as gee_id metadata. station_id, gee_id, lat, lon, date and datetime_utc are never model features.
3. Cleaning rules and rows removed: missing PM2.5: 0; PM2.5 <= 0: 727; PM2.5 >= 500: 8; exact duplicates: 0; date+lat+lon+station duplicates: 0; same sensor and same day with different GEE ids: 4 rows (2 groups, MOLINA DE SEGURA and SAN BASILIO), both rows of each group dropped to avoid an arbitrary pick; unparseable dates: 0. Result: 40,979 rows.
4. Nothing learned from data is done in cleaning: NO2, CO and O3 stay missing as they are (structurally missing 2016 to 2018), and no imputation, scaling or outlier treatment is applied. All of that belongs in the training-fold Pipeline.
5. Final test period (proposal, team confirms): 2025-01-01 to 2025-12-31 (8,149 rows, 57 stations), locked in data/final_test/final_test.parquet. Development data = rows before 2025 (31,282 rows, 70 stations, 2016-11-25 to 2024-12-31). Jan to Apr 2026 (1,548 rows) is also locked away, because it is later than the test year.

## Go / no-go
- Temporal CV: GO. All dates parse, datetime_utc is midnight and matches date, 0 duplicate timestamps after cleaning.
- Spatial CV grouped by station_id: GO. No sensor moves more than 0.05 degrees, no GEE id is shared between sensors, and the 4 sensors with two GEE ids stay in one group.
- No target-derived column was found (by name or by |corr| > 0.95).

## Open points for the team
- NO2, CO, O3 are missing for all of 2016 and 2017 and most of 2018, then complete from 2019.
- Sparse early years: 111 rows in 2016, 252 in 2018.
- 15 sensors have fewer than 100 valid days; 13 of the 57 test-year stations never appear in development data. Retained stations are decided Thursday.
- The Kaggle description lists 125 PM2.5 sensors and 100 GEE ids; the combined file has 94 sensor names and 98 GEE ids.
- The description attributes NO2, CO and O3 gaps (about 12%) to cloud cover, but in the combined file the missingness is entirely before 2019, so it looks structural.
