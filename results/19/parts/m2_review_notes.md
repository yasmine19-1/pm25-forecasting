# review follow-up

## Predictors between stations (§5.1)

Figures: `m2_station_<variable>.png` (one per predictor, stations from south
to north), `m2_station_heatmap.png` (all predictors at once),
`m2_station_eta2.png` (summary).

eta² = share of a predictor's variance explained by the station alone:

| predictor | eta² |
|---|---|
| pressure_mb | 0.90 |
| no2 | 0.19 |
| wind_v | 0.10 |
| co | 0.09 |
| temperature_celsius | 0.05 |
| o3 | 0.04 |
| aod | 0.03 |
| wind_u | 0.03 |

- **Pressure** is almost entirely a station property (eta² 0.90):
  it reflects altitude (lowest median at GRANADA - NORTE, highest at FR02013). It works
  as a proxy for elevation, which is fine physically, but it also lets a model
  recognise a station: worth remembering when reading spatial-CV results.
- **NO₂** differs clearly between stations (eta² 0.19): highest
  median at La Sagrera, lowest at ALCANTARILLA. Urban and industrial sites stand out.
- **AOD, O₃, temperature and wind** barely depend on the station (eta² under
  0.1): most of their variation is day to day, shared across the region.
  Satellite pixels are coarse (several stations share one pixel), so these
  columns carry little information on *where* PM2.5 is high, mostly on *when*.
- Every predictor still differs significantly between stations
  (Kruskal-Wallis p < 0.001), but with 31,000 rows that test is always
  significant: eta² is the useful measure.

## Flagged extreme values: corroborated episode candidates or data errors? (§5.1, §5.3.3)

All 268 PM2.5 values above Q3 + 3×IQR were checked one by one
(`m2_extremes_investigation.csv`, one row per value, with the evidence and the
provisional label). Labels are corroboration-based. Evidence used: stations within 30 km the same day, the
station's own values the day before and after, the jump against the
station's ±3-day median, AOD, and repeated exact values. A neighbour or an
adjacent day only "confirms" a value if it is above its own 90th percentile
and at least a fifth of the extreme.

| verdict | n | median PM2.5 |
|---|---|---|
| corroborated episode candidate | 130 | 39.6 |
| plausible local event | 63 | 46.4 |
| possible data error | 35 | 112.0 |
| undetermined | 32 | 39.2 |
| suspect, unconfirmed | 8 | 47.6 |

- **Corroborated episode candidates (130)** are moderate values (median
  40 µg/m³) shared with nearby stations.
  71% happen in November to February, mostly at urban stations
  (VALÈNCIA-PISTA DE SILLA (10), La Sagrera (8), VALÈNCIA-POLITÈCNIC (8), HOSTELERÍA (7)), consistent with winter inversions, heating and traffic. These
  observations remain in the development set.
- **Possible data errors (35)** look very
  different: much higher values (median 112 µg/m³),
  a one-day spike (5 to 76 times the station's usual level) while
  neighbours within 30 km stay normal and AOD is ordinary. They
  concentrate at a few stations: CENCA1 (7), LOS BARRIOS (7), HOSTELERÍA (3), ALCORA (3). 6 of the 10 highest
  values in the dataset fall in this group, including the 485 µg/m³ at CENCA1
  on 2024-09-04 (neighbour 14.7 µg/m³ the same day).
- **Plausible local events (63)**: the
  station stays high for two days or more while neighbours do not. A sensor
  fault can last two days, but so can a local source. Example: in November
  2021 Los Barrios and Campamento (Campo de Gibraltar, an industrial area)
  each stay above 200 µg/m³ for two days, and on 23 November a neighbour
  reaches 123 µg/m³. A local industrial episode is possible, but the
  evidence is mixed. These are kept.
- **Suspect, unconfirmed (8)** and
  **undetermined (32)**: no neighbour to compare
  with, or not enough evidence. Kept.
- Case studies: `m2_extreme_case_1.png` (corroborated episode candidate),
  `m2_extreme_case_2.png` (local event), `m2_extreme_case_3.png` (data error).

### Retention
All screened extremes remain in the development set. The review uses
development-wide context and can include observations after the event date.
The labels are descriptive and are excluded from modeling inputs.
