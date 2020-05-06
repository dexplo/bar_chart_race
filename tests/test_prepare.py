import pandas as pd
import bar_chart_race as bcr


class TestPrepareWideData:

    df_wide = pd.read_csv('../data/test_data/covid_test.csv', index_col='date', parse_dates=['date'])

    def test_prepare_wide_data(self):
        df_wide_values, df_wide_ranks = bcr.prepare_wide_data(self.df_wide)
        df_wide_values_ans = pd.read_csv('../data/test_data/covid_test_values.csv', 
                                         index_col='date', parse_dates=['date'])
        df_wide_ranks_ans = pd.read_csv('../data/test_data/covid_test_ranks.csv', 
                                        index_col='date', parse_dates=['date'])
        pd.testing.assert_frame_equal(df_wide_values, df_wide_values_ans)
        pd.testing.assert_frame_equal(df_wide_ranks, df_wide_ranks_ans)