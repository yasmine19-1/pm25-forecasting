"""Reproduce development-only matching and temporal decision evidence."""
import numpy as np
import pandas as pd
from src import config
from src.common import parse_dates


def past_pairs(g, column, lag):
    """Pair current target with past values only inside uninterrupted daily runs."""
    g = g.sort_values('date')
    same_run = g.date.diff().dt.days.ne(1).cumsum()
    past = g.groupby(same_run)[column].shift(lag)
    return pd.DataFrame({'current': g.pm25, 'past': past}).dropna()


def correlations(pairs):
    if len(pairs) < 3 or pairs.nunique().min() < 2:
        return np.nan, np.nan
    return pairs.current.corr(pairs.past), pairs.current.corr(pairs.past, method='spearman')


def main():
    df = pd.read_parquet(config.INTERIM / 'clean_dev.parquet')
    df.date = parse_dates(df.date)
    assert len(df) == 31282 and df.station_id.nunique() == 70
    assert df.date.notna().all() and df.date.eq(df.date.dt.normalize()).all()
    assert df.date.lt(config.TEST_START).all() and not df.duplicated(['station_id','date']).any()
    # Read source metadata only; never run the raw-data diagnostic modules.
    a = pd.read_csv(config.RAW / 'pm25.csv', usecols=['sensor_name','date','latitude','longitude'])
    b = pd.read_csv(config.RAW / 'gee_features_daily.csv', usecols=['id','date','lat','lon'])
    for x in (a,b):
        x['date'] = parse_dates(x.date)
    a = a.loc[a.date.lt(config.TEST_START)].copy()
    b = b.loc[b.date.lt(config.TEST_START)].copy()
    a = a.rename(columns={'sensor_name':'station_id','latitude':'lat','longitude':'lon'})
    b = b.rename(columns={'id':'gee_id','lat':'gee_lat','lon':'gee_lon'})
    keys = ['station_id','date','lat','lon']
    source_present = pd.MultiIndex.from_frame(df[keys]).isin(pd.MultiIndex.from_frame(a[keys]))
    if b.duplicated(['gee_id','date']).any():
        raise ValueError('Ambiguous source GEE ID/date metadata; inspect before generating evidence')
    merged = df[['station_id','date','gee_id','lat','lon']].merge(b, on=['gee_id','date'], how='left', validate='many_to_one', indicator=True)
    merged['source_station_metadata_present'] = source_present
    merged['assigned_distance_degrees'] = np.hypot(merged.lat-merged.gee_lat, merged.lon-merged.gee_lon)
    # Compare assigned point to all source GEE locations available on that date.
    merged['nearest_distance_degrees'] = np.nan
    for day, idx in merged.groupby('date').groups.items():
        candidates = b.loc[b.date.eq(day), ['gee_lat','gee_lon']].drop_duplicates().to_numpy()
        if len(candidates):
            points = merged.loc[idx,['lat','lon']].to_numpy()
            distances = np.sqrt(((points[:,None,:]-candidates[None,:,:])**2).sum(axis=2))
            merged.loc[idx,'nearest_distance_degrees'] = distances.min(axis=1)
    merged['assigned_is_nearest'] = np.isclose(merged.assigned_distance_degrees, merged.nearest_distance_degrees, atol=1e-10, rtol=0)
    matching = merged.groupby('station_id').agg(n_obs=('date','size'),
        source_station_rows_found=('source_station_metadata_present','sum'),
        source_gee_rows_found=('_merge',lambda s: s.eq('both').sum()),
        nearest_assigned_rows=('assigned_is_nearest','sum'),
        max_assigned_distance_degrees=('assigned_distance_degrees','max'))
    matching.to_csv(config.PARTS / 'thursday_matching.csv')
    eligibility = pd.read_csv(config.INTERIM / 'station_eligibility.csv')
    decision = eligibility.merge(matching.reset_index(), on='station_id', validate='one_to_one', suffixes=('', '_match'))
    if len(decision) != 70 or not decision.integrity_ok.all() or not decision.n_obs.eq(decision.n_obs_match).all():
        raise ValueError('Station integrity evidence changed; re-review the decision')
    for column in ('source_station_rows_found', 'source_gee_rows_found', 'nearest_assigned_rows'):
        if not decision[column].eq(decision.n_obs).all():
            raise ValueError('Source matching evidence changed; re-review before retaining stations')
    retained = decision[['station_id', 'n_obs', 'in_ge100_days']].copy()
    retained['retain'] = True
    retained['reason'] = 'Passed metadata integrity and source matching; no evidence-based exclusion'
    retained['limitation'] = decision.integrity_note.where(decision.needs_review,
        'Coverage and history length remain analysis-specific limitations')
    retained['sensitivity_note'] = '>=100 days is a sensitivity cohort only, not a retention rule'
    retained.to_csv(config.PARTS / 'retained_stations.csv', index=False)
    rows=[]
    variables=['pm25','aod','no2','co','o3','temperature_celsius','pressure_mb','wind_u','wind_v']
    for station,g in df.groupby('station_id'):
        for variable in variables:
            for lag in (1,2,3,7):
                pairs=past_pairs(g,variable,lag)
                pearson,spearman=correlations(pairs)
                rows.append((station,variable,lag,len(pairs),pearson,spearman))
    temporal=pd.DataFrame(rows,columns=['station_id','past_variable','lag_days','n_pairs','pearson','spearman'])
    temporal.to_csv(config.PARTS / 'thursday_lag_diagnostics.csv',index=False)
    # Year-specific target persistence checks show whether pooled results hide drift.
    yearly=[]
    for station,g in df.groupby('station_id'):
        pairs=past_pairs(g,'pm25',1)
        pairs['year']=df.loc[pairs.index,'date'].dt.year
        for year,p in pairs.groupby('year'):
            p=p[['current','past']]
            pearson,spearman=correlations(p)
            yearly.append((station,year,len(p),pearson,spearman))
    pd.DataFrame(yearly,columns=['station_id','year','n_pairs','pearson','spearman']).to_csv(config.PARTS / 'thursday_persistence_by_year.csv',index=False)
    # Descriptive seasonal/drift summaries; varying station composition is explicit.
    profile=[]
    for unit,key in [('year',df.date.dt.year),('month',df.date.dt.month),('weekday',df.date.dt.day_of_week)]:
        for value,g in df.groupby(key):
            for var in variables:
                s=g[var].dropna()
                profile.append((unit,value,var,len(s),g.station_id.nunique(),s.median(),s.quantile(.75)-s.quantile(.25),
                                s.gt(25).mean() if var == 'pm25' else np.nan,
                                s.gt(50).mean() if var == 'pm25' else np.nan))
    pd.DataFrame(profile,columns=['period','value','variable','n_obs','n_stations','median','iqr',
                                 'share_above_25','share_above_50']).to_csv(config.PARTS / 'thursday_temporal_profiles.csv',index=False)
    summary=[]
    for (var,lag),g in temporal.groupby(['past_variable','lag_days']):
        usable=g.spearman.notna()
        summary.append(['temporal_gate',var,f'lag_{lag}d','n_pairs',g.n_pairs.sum(),'Past values within continuous daily runs only'])
        summary.append(['temporal_gate',var,f'lag_{lag}d','median_station_spearman',g.spearman.median(),'Descriptive; not significance or predictive skill'])
        summary.append(['temporal_gate',var,f'lag_{lag}d','stations_estimable',int(usable.sum()),'At least 3 pairs and both variables nonconstant; mathematical computability only'])
        summary.append(['temporal_gate',var,f'lag_{lag}d','stations_positive',int(g.spearman.gt(0).sum()),'No inferential threshold or station exclusion'])
    pd.DataFrame(summary,columns=['section','variable','group','metric','value','note']).to_csv(config.PARTS / 'thursday_temporal_summary.csv',index=False)
    print('Matching totals:',matching[['n_obs','source_station_rows_found','source_gee_rows_found','nearest_assigned_rows']].sum().to_dict())
    print('Maximum assigned matching distance (degrees):',matching.max_assigned_distance_degrees.max())
    print(temporal.loc[temporal.past_variable.eq('pm25')].groupby('lag_days').agg(n_pairs=('n_pairs','sum'),median_spearman=('spearman','median'),estimable=('spearman','count'),positive=('spearman',lambda s:s.gt(0).sum())).to_string())
    print('Wrote development-only evidence. No datasets or locked test files modified.')


if __name__=='__main__':
    main()
