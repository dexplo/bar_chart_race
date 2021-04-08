import pytest
import bar_chart_race._utils as utils
from bar_chart_race import load_dataset, bar_chart_race

df = load_dataset('baseball')
df = df.iloc[-20:-16]


def test_threshold():
    filtered_df = utils.filter_threshold(df, 60)
    assert len(filtered_df) == 1

    filtered_df = utils.filter_threshold(df, 0)
    assert len(filtered_df) == 4

    filtered_df = utils.filter_threshold(df, 50)
    assert filtered_df.iloc[0]['hr'] > 50