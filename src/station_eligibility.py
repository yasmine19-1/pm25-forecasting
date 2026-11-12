"""Create a metadata-only evidence table; never apply station exclusions."""
import json
import re

import numpy as np
import pandas as pd

from src import config
from src.common import parse_dates
from src.gap_report import station_metrics

METRICS = ['calendar_coverage_percentage', 'longest_continuous_run_days',
           'maximum_gap_days'] + [f'usable_rows_with_{k}d_contiguous_history' for k in (1, 2, 3, 7)]


def check(label, passed, detail):
    print(f"{'PASS' if passed else 'FAIL'}: {label}: {detail}")
    return bool(passed)


def coordinate_evidence(g, bounds):
    g = g.sort_values('date').copy()
    lat = pd.to_numeric(g.lat, errors='coerce')
    lon = pd.to_numeric(g.lon, errors='coerce')
    missing = g[['lat', 'lon']].isna().any(axis=1)
    valid = lat.between(-90, 90) & lon.between(-180, 180)
    invalid = ~missing & ~valid
    pairs = g.loc[~missing, ['lat', 'lon']].drop_duplicates()
    conflict_dates = g.loc[~missing].groupby('date').apply(
        lambda x: len(x[['lat', 'lon']].drop_duplicates()), include_groups=False)
    conflicts = conflict_dates[conflict_dates > 1].index
    # Compare adjacent dated observations only; never bridge an invalid coordinate.
    ok = valid & ~g.date.isin(conflicts)
    adjacent = ok & ok.shift(fill_value=False) & g.date.gt(g.date.shift())
    a, b = np.radians(lat), np.radians(lon)
    h = np.sin((a - a.shift()) / 2)**2 + np.cos(a) * np.cos(a.shift()) * np.sin((b - b.shift()) / 2)**2
    distances = (2 * 6371.0088 * np.arcsin(np.sqrt(h.clip(0, 1)))).where(adjacent)
    if distances.notna().any():
        idx = distances.idxmax()
        distance, start, end = distances.loc[idx], g.date.shift().loc[idx], g.date.loc[idx]
    else:
        distance, start, end = np.nan, pd.NaT, pd.NaT
    outside = 0
    if bounds is not None:
        lo_lat, hi_lat, lo_lon, hi_lon = bounds
        # The audit bounds are rounded to four decimal places, not regional limits.
        outside = int((valid & ~(lat.between(lo_lat - 0.00005, hi_lat + 0.00005)
                                & lon.between(lo_lon - 0.00005, hi_lon + 0.00005))).sum())
    return dict(n_coordinate_pairs=len(pairs), n_missing_coordinates=int(missing.sum()),
                n_invalid_coordinates=int(invalid.sum()), n_conflicting_coordinate_dates=len(conflicts),
                n_outside_audit_envelope=outside if bounds is not None else pd.NA,
                max_coordinate_change_km=distance,
                coordinate_change_from_date=start.strftime('%Y-%m-%d') if pd.notna(start) else '',
                coordinate_change_to_date=end.strftime('%Y-%m-%d') if pd.notna(end) else '')


def test_presence():
    path = config.FINAL_TEST / 'final_test.parquet'
    try:
        if config.TEST_START != '2025-01-01' or config.TEST_END != '2025-12-31':
            raise ValueError('Configured holdout dates differ from 2025')
        # Path and column names follow src/clean.py; no target or predictor is read.
        t = pd.read_parquet(path, columns=['station_id', 'date'])
        dates = parse_dates(t.date)
        if not (dates.notna().all() and dates.between(config.TEST_START, config.TEST_END).all()
                and dates.eq(dates.dt.normalize()).all()
                and t.station_id.notna().all() and t.station_id.astype(str).str.strip().ne('').all()):
            raise ValueError('Holdout identifier/date validation failed')
        if t.assign(date=dates).duplicated(['station_id', 'date']).any():
            raise ValueError('Duplicate holdout station-days')
        return t.groupby('station_id').size(), 'available: station_id/date projection only'
    except Exception as exc:
        return None, f'unavailable: {type(exc).__name__}: {exc}'


def main():
    mapping = pd.read_csv(config.PARTS / 'column_mapping.csv')
    df = pd.read_parquet(config.INTERIM / 'clean_dev.parquet',
                         columns=['station_id', 'date', 'gee_id', 'lat', 'lon'])
    df['date'] = parse_dates(df.date)
    original = pd.read_csv(config.PARTS / 'gap_report_enhanced.csv')
    checks = []
    dates_ok = (df.date.notna().all() and df.date.eq(df.date.dt.normalize()).all()
                and df.date.between('2016-01-01', '2024-12-31').all())
    checks.append(check('Development dates', dates_ok, 'parsed UTC-naive, midnight, 2016–2024'))
    checks.append(check('Identifier mapping',
                        mapping.loc[mapping.clean_name.eq('station_id'), 'raw_name'].tolist() == ['sensor_name'],
                        'development station_id -> report station_id; raw provenance sensor_name'))
    checks.append(check('1. Development totals', df.station_id.nunique(dropna=False) == 70 and len(df) == 31282,
                        f'{df.station_id.nunique(dropna=False)} stations, {len(df)} observations'))
    if not dates_ok:
        for label in ('2. Sensitivity cohort', '3. Contiguous histories', '4. Longest run'):
            checks.append(check(label, False, 'not evaluable: invalid development dates'))
        checks.append(check('5. Duplicate station-days', not df.duplicated(['station_id', 'date']).any(),
                            str(int(df.duplicated(['station_id', 'date']).sum()))))
        raise SystemExit('Stopped without writing reports: date validation failed.')
    computed = pd.DataFrame([station_metrics(s, g.date) for s, g in df.groupby('station_id', dropna=False)])
    cohort = computed[computed.unique_dates >= 100]
    checks.append(check('2. Sensitivity cohort', len(cohort) == 63 and cohort.number_of_rows.sum() == 30963,
                        f'{len(cohort)} stations, {cohort.number_of_rows.sum()} observations'))
    histories = [int(cohort[f'usable_rows_with_{k}d_contiguous_history'].sum()) for k in (1, 3, 7)]
    checks.append(check('3. Contiguous histories (1 / 3 / 7 days)', histories == [20264, 10060, 3350], str(histories)))
    longest = int(computed.longest_continuous_run_days.max())
    checks.append(check('4. Longest uninterrupted run', longest == 32, f'{longest} days'))
    duplicates = int(df.duplicated(['station_id', 'date']).sum())
    checks.append(check('5. Duplicate station-days', duplicates == 0, f'{duplicates} duplicates'))
    join_ok = (not original.station_id.duplicated().any() and not computed.station_id.duplicated().any()
               and set(original.station_id) == set(computed.station_id))
    checks.append(check('Gap-report join', join_ok, 'complete station sets and one-to-one keys'))
    metrics_ok = False
    if join_ok:
        x = computed.set_index('station_id').sort_index()
        y = original.set_index('station_id').sort_index()
        metrics_ok = all(np.allclose(x[c], y[c], atol=0.000051, rtol=0, equal_nan=True)
                         if pd.api.types.is_numeric_dtype(x[c]) else x[c].equals(y[c]) for c in x.columns)
    checks.append(check('Existing gap metrics', metrics_ok, 'all columns recomputed with existing station_metrics; 4-decimal tolerance'))
    if not all(checks):
        raise SystemExit('Stopped without writing reports: acceptance or consistency check failed.')

    audit = pd.read_csv(config.PARTS / 'audit.csv')
    bounds = None
    try:
        values = audit.set_index('variable').value
        bounds = tuple(float(v) for key in ('lat_min_max', 'lon_min_max')
                       for v in re.fullmatch(r'([-\d.]+) to ([-\d.]+)', str(values[key])).groups())
    except (KeyError, AttributeError, ValueError):
        pass
    df['gee_id'] = df.gee_id.mask(df.gee_id.astype('string').str.strip().eq(''))
    sharing = df.dropna(subset=['gee_id']).groupby('gee_id').station_id.nunique()
    evidence = []
    for station, g in df.groupby('station_id', dropna=False):
        row = {'station_id': station, **coordinate_evidence(g, bounds)}
        ids = g.gee_id.dropna().unique()
        row.update(n_gee_ids=len(ids), n_missing_gee_ids=int(g.gee_id.isna().sum()),
                   n_stations_sharing_gee_id=int(sharing.reindex(ids).max()) if len(ids) else pd.NA,
                   n_missing_or_blank_station_ids=int((g.station_id.isna() | g.station_id.astype('string').str.strip().eq('')).sum()))
        failures, reviews = [], []
        for field, label in [('n_missing_coordinates', 'missing coordinates'),
                             ('n_invalid_coordinates', 'invalid coordinates'),
                             ('n_conflicting_coordinate_dates', 'conflicting coordinates on same date'),
                             ('n_missing_or_blank_station_ids', 'missing/blank station identifier')]:
            if row[field]:
                failures.append(f'{label}: {row[field]}')
        if len(ids) > 1:
            reviews.append(f'{len(ids)} GEE IDs')
        if row['n_coordinate_pairs'] > 1:
            reviews.append(f"{row['n_coordinate_pairs']} coordinate pairs; change {row['max_coordinate_change_km']:.6f} km")
        if len(ids) and row['n_stations_sharing_gee_id'] > 1:
            reviews.append(f"GEE ID shared by up to {row['n_stations_sharing_gee_id']} stations")
        if row['n_missing_gee_ids']:
            reviews.append(f"missing GEE IDs: {row['n_missing_gee_ids']}")
        if bounds is None:
            reviews.append('documented audit envelope unavailable')
        elif row['n_outside_audit_envelope']:
            reviews.append(f"outside documented audit envelope: {row['n_outside_audit_envelope']}")
        row.update(integrity_ok=not failures, needs_review=bool(failures or reviews),
                   integrity_note='; '.join(failures + reviews) or 'Passed defined metadata integrity checks')
        for year in range(2016, 2025):
            row[f'obs_{year}'] = int(g.date.dt.year.eq(year).sum())
        evidence.append(row)
    report = original[['station_id', 'number_of_rows', 'unique_dates', 'first_date', 'last_date'] + METRICS].rename(
        columns={'number_of_rows': 'n_obs', 'unique_dates': 'n_observed_days'})
    report = report.merge(pd.DataFrame(evidence), on='station_id', validate='one_to_one', how='left')
    presence, presence_note = test_presence()
    report['obs_2025'] = report.station_id.map(presence).fillna(0).astype('Int64') if presence is not None else pd.NA
    report['in_ge100_days'] = report.n_observed_days >= 100
    ended = report.loc[report.last_date < '2024-01-01', 'station_id'].tolist()
    zero = report.loc[report.obs_2025.eq(0), 'station_id'].tolist() if presence is not None else []
    checks.append(check('Final evidence join and annual counts',
                        len(report) == 70 and report.station_id.is_unique and report.integrity_ok.notna().all()
                        and report[[f'obs_{y}' for y in range(2016, 2025)]].sum(axis=1).eq(report.n_obs).all(),
                        '70 unique rows; annual counts sum to n_obs'))
    if not all(checks):
        raise SystemExit('Stopped without writing reports: final evidence validation failed.')
    rows = []
    def add(variable, group, metric, value, note=''):
        rows.append(['station_eligibility', variable, group, metric, value, note])
    add('development', 'all', 'stations', len(report))
    add('development', 'all', 'observations', int(report.n_obs.sum()))
    add('integrity_ok', 'all', 'stations', int(report.integrity_ok.sum()), 'Defined metadata checks only; not a retention decision')
    add('needs_review', 'all', 'stations', int(report.needs_review.sum()))
    add('last_date_before_2024', 'all', 'stations', len(ended), json.dumps(ended, ensure_ascii=False))
    add('zero_obs_2025', 'development_stations', 'stations', len(zero) if presence is not None else 'unavailable',
        json.dumps(zero, ensure_ascii=False) if presence is not None else presence_note)
    for metric, value in [('stations', int(report.in_ge100_days.sum())),
                          ('observations', int(report.loc[report.in_ge100_days, 'n_obs'].sum()))]:
        add('in_ge100_days', 'sensitivity', metric, value, 'sensitivity cohort only, not a retention rule')
    summary = pd.DataFrame(rows, columns=['section', 'variable', 'group', 'metric', 'value', 'note'])
    notes = f'''# Station eligibility evidence

Reproduce from the repository root: `.venv/bin/python -m src.station_eligibility`.
No station exclusions have been applied. No source datasets or existing gap reports
are written. No folds or models are generated. Temporal-analysis eligibility is a
separate consideration. The brief supplies no minimum-observation threshold;
`in_ge100_days` means **sensitivity cohort only, not a retention rule**.

## Inputs and mapping
Development projection: station_id, date, gee_id, lat, lon only. Saved column_mapping.csv
and cleaning code confirm raw sensor_name -> development station_id -> report station_id;
raw id -> gee_id, latitude -> lat, longitude -> lon. No target/predictor values are read.
The locked file is config.FINAL_TEST / final_test.parquet, as written by src/clean.py.
Its only projected columns are station_id and date. obs_2025 is a count for each
DEVELOPMENT station, not an evaluation or a list of test-only stations. Status: {presence_note}.
If this check is unavailable, obs_2025 is written as the literal 'unavailable', never zero.
No 2026 file is opened. The configured test dates must equal calendar 2025.

## Dates and gap metrics
Use src.common.parse_dates (UTC converted to timezone-naive), then verify midnight;
non-midnight or invalid dates are rejected, not rounded into passing observations.
Import src.gap_report.station_metrics; recompute and compare EVERY existing metric
against gap_report_enhanced.csv before writing. Joins have complete, one-to-one keys.
number_of_rows -> n_obs; unique_dates -> n_observed_days; requested coverage, run,
gap and contiguous-history columns retain their existing names and definitions.
Coverage is observed days / inclusive first-to-last span * 100, not whole-period
coverage. Maximum gap is elapsed days between successive observations (includes
endpoints' separation; missing days = gap minus one). Runs are uninterrupted daily
sequences; a run of L days contributes max(L-K, 0) rows with K preceding consecutive
days. No interpolation or gap bridging. Counts are metadata availability, not a
claim that model inputs exist or that seasonality can be estimated. Annual counts
use observation dates. Sparse data do not imply sensor malfunction.

## Integrity definitions
integrity_ok requires nonblank station identity, valid normalized development dates,
no duplicate station/date pairs, complete numeric coordinates in latitude [-90,90]
and longitude [-180,180], and no conflicting same-date coordinate pairs. Date and
duplicate checks are global acceptance gates: a failure prevents report generation.
Station-specific coordinate/identity failures are reported, not removed.
needs_review flags those failures plus multiple coordinate pairs, multiple GEE IDs,
shared GEE IDs, missing GEE IDs, and deviations from the documented audit envelope.
It does NOT flag observation counts or coverage. integrity_ok can be True while
needs_review is True. Neither flag authorizes exclusion or proves leakage.

Distinct coordinate pairs use exact stored values, both nonmissing; invalid nonmissing
pairs remain counted and are separately flagged. Missing coordinates means either
component is null. Invalid means a nonmissing pair is nonnumeric, nonfinite or outside
valid global ranges. Blank GEE IDs are treated as missing, without rewriting input.
n_stations_sharing_gee_id is the maximum distinct-station count for any assigned
nonmissing ID, including this station; unavailable when no ID is assigned.

Distance uses the haversine formula with mean Earth radius 6371.0088 km; longitude
and latitude are converted to radians and the numerical haversine is clipped to [0,1].
Only successive dated observations with valid coordinates at both endpoints are
compared; invalid endpoints are not bridged. This is recorded-position change,
not physical travel or speed. The largest distance and both dates are saved; ties
use the earliest pair. Constant coordinates yield zero with the first adjacent pair.
No eligible pair yields unavailable distance and blank dates. Same-date conflicting
coordinates are counted and excluded from distance comparisons, never resolved by
picking one; the duplicate acceptance gate prevents a final report in that case.

## Existing audit and limitations
The existing audit.csv and the shared Phase 1 methodology document the station/GEE reliability
audit: four multi-ID stations (LA ALJORRA, MOLINA DE SEGURA, MOMPEAN, SAN BASILIO),
no shared GEE IDs, and coordinate spread below 0.05 degrees. Current development
metadata are checked independently; historical whole-dataset counts are not used
as development acceptance criteria. The audit's geographic extrema are {bounds}.
They are observed dataset bounds, NOT a regional validity boundary. Compare with
0.00005-degree rounding allowance because audit.csv records four decimals; deviations
trigger review only. No new geographic boundary or movement cutoff is invented.
The matching diagnostic code uses a 0.25-degree nearest-point rule, but no saved
per-station nearest-match evidence was found in results. The development table has
only one coordinate pair per observation, not independent station/GEE locations.
Thus nearest-GEE matching cannot be independently revalidated here; this is a global
scope limitation, not evidence against every station. No raw matching script is run.
The four documented multi-ID stations remain explicit review cases.

## Results
All five required acceptance checks passed before report generation.
Integrity checks passed: {int(report.integrity_ok.sum())}/70. Review flags: {int(report.needs_review.sum())}.
Stations ending before 2024: {len(ended)}: {json.dumps(ended, ensure_ascii=False)}.
Development stations with zero 2025 observations: {len(zero) if presence is not None else 'unavailable'}:
{json.dumps(zero, ensure_ascii=False) if presence is not None else presence_note}.
'''
    config.PARTS.mkdir(parents=True, exist_ok=True)
    report.to_csv(config.INTERIM / 'station_eligibility.csv', index=False, na_rep='unavailable', float_format='%.6f')
    summary.to_csv(config.PARTS / 'station_eligibility_summary.csv', index=False)
    (config.PARTS / 'station_eligibility_notes.md').write_text(notes, encoding='utf-8')
    print(report.loc[~report.integrity_ok | report.needs_review, ['station_id', 'integrity_ok', 'integrity_note',
        'coordinate_change_from_date', 'coordinate_change_to_date']].to_string(index=False))
    print('Ending before 2024:', json.dumps(ended, ensure_ascii=False))
    print('Zero observations in 2025:', json.dumps(zero, ensure_ascii=False) if presence is not None else presence_note)
    print(f'Wrote evidence table, summary and notes; 2025 metadata {presence_note}.')


if __name__ == '__main__':
    main()
