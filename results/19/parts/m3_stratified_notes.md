# Member 3 - PM2.5 and AOD (auto-generated, check the numbers before pasting)

Overall (n = 31277): Pearson 0.07, Spearman 0.15, slope 6.5 ug/m3 per AOD unit.

By season (Spearman): DJF 0.24, MAM 0.18, JJA 0.15, SON 0.20. The link is strongest in DJF and weakest in JJA.

By station (67 stations with at least 30 pairs, 3 skipped): median Spearman 0.18 (IQR 0.09 to 0.26); 96% of the stations have a positive link.

temperature_celsius: slope of PM2.5 on AOD in the low/middle/high tertile = 10.9 / 11.1 / 5.4; standardised AOD x temperature_celsius coefficient +0.03 (steeper PM2.5-AOD slope when temperature_celsius is higher).
pressure_mb: slope of PM2.5 on AOD in the low/middle/high tertile = 9.6 / 10.2 / 1.1; standardised AOD x pressure_mb coefficient -0.27 (flatter PM2.5-AOD slope when pressure_mb is higher).
wind_speed: slope of PM2.5 on AOD in the low/middle/high tertile = 5.1 / 2.7 / 10.4; standardised AOD x wind_speed coefficient +0.18 (steeper PM2.5-AOD slope when wind_speed is higher).

Reading guide: these are associations on development data, not causal effects. AOD is a column-integrated satellite value while PM2.5 is measured at the surface, so boundary-layer conditions (temperature, pressure, wind) are expected to change the link.
