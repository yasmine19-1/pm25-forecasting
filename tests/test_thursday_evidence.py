"""Verify that temporal evidence never treats missing days as consecutive."""
import unittest
import pandas as pd
from src.thursday_evidence import past_pairs, correlations


class TemporalEvidenceTests(unittest.TestCase):
    def test_gap_and_past_direction(self):
        g=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-02','2024-01-05','2024-01-06']),
                        'pm25':[10,20,30,40], 'aod':[1,2,3,4]})
        pairs=past_pairs(g,'aod',1)
        self.assertEqual(pairs.to_dict('list'),{'current':[20,40],'past':[1.,3.]})
        self.assertTrue(past_pairs(g,'aod',2).empty)

    def test_missing_predictor_does_not_shift_dates(self):
        g=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=4),
                        'pm25':[1,2,3,4], 'aod':[10,None,30,40]})
        pairs=past_pairs(g,'aod',1)
        self.assertEqual(pairs.current.tolist(),[2,4])
        self.assertEqual(pairs.past.tolist(),[10,30])

    def test_unestimable_correlation_stays_unavailable(self):
        p=pd.DataFrame({'current':[1,2,3],'past':[5,5,5]})
        self.assertTrue(all(pd.isna(x) for x in correlations(p)))


if __name__=='__main__':
    unittest.main()
