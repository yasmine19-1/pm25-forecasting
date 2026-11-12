"""Check metadata boundaries and non-destructive failure behavior."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import config
from src.station_eligibility import coordinate_evidence, main, test_presence as presence


class StationEvidenceTests(unittest.TestCase):
    def test_distance_and_invalid_endpoint(self):
        frame = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3),
                              'lat': [0., 0., 0.], 'lon': [0., 1., 2.]})
        result = coordinate_evidence(frame, None)
        self.assertAlmostEqual(result['max_coordinate_change_km'], 111.195080, places=5)
        self.assertEqual(result['coordinate_change_from_date'], '2024-01-01')
        frame.loc[1, 'lat'] = 100
        result = coordinate_evidence(frame, None)
        self.assertEqual(result['n_invalid_coordinates'], 1)
        self.assertTrue(pd.isna(result['max_coordinate_change_km']))

    def test_conflicting_same_date_is_not_arbitrarily_selected(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02']),
                              'lat': [0., 1., 2.], 'lon': [0., 0., 0.]})
        result = coordinate_evidence(frame, None)
        self.assertEqual(result['n_conflicting_coordinate_dates'], 1)
        self.assertTrue(pd.isna(result['max_coordinate_change_km']))

    def test_unavailable_test_metadata_and_projection(self):
        with patch('src.station_eligibility.pd.read_parquet', side_effect=OSError('not accessible')) as read:
            counts, note = presence()
        self.assertIsNone(counts)
        self.assertTrue(note.startswith('unavailable:'))
        read.assert_called_once_with(config.FINAL_TEST / 'final_test.parquet', columns=['station_id', 'date'])

    def test_failed_acceptance_writes_nothing(self):
        cols = ['station_id', 'date', 'gee_id', 'lat', 'lon']
        frame = pd.read_parquet(config.INTERIM / 'clean_dev.parquet', columns=cols)
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            with patch.object(config, 'INTERIM', output), patch('src.station_eligibility.pd.read_parquet', return_value=frame), contextlib.redirect_stdout(io.StringIO()) as log:
                with self.assertRaises(SystemExit):
                    main()
            self.assertIn('FAIL: 1.', log.getvalue())
            self.assertIn('FAIL: 5.', log.getvalue())
            self.assertEqual(list(output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
