import pytest
import matplotlib.pyplot as plt
from bar_chart_race import load_dataset, bar_chart_race


df = load_dataset('covid19')
df = df.iloc[-20:-16]
df1 = df.reset_index(drop=True)


class TestSimpleBC:

    def test_defaults(self):
        bar_chart_race(df)
        bar_chart_race(df, orientation='v')

    def test_sort(self):
        bar_chart_race(df, sort='asc')
        bar_chart_race(df, orientation='v', sort='asc')

    def test_nbars(self):
        bar_chart_race(df, sort='desc', n_bars=8)
        bar_chart_race(df, orientation='v', sort='desc', n_bars=8)

    def test_fixed_order(self):
        bar_chart_race(df, sort='asc', n_bars=8, fixed_order=True)
        bar_chart_race(df, fixed_order=['Iran', 'USA', 'Italy', 'Spain'])

    def test_fixed_max(self):
        bar_chart_race(df, fixed_max=True)

    def test_steps_per_period(self):
        bar_chart_race(df, sort='asc', steps_per_period=2)
        bar_chart_race(df, sort='asc', steps_per_period=30)

    def test_interpolate_period(self):
        bar_chart_race(df, interpolate_period=True, n_bars=8)

    def test_bar_size(self):
        bar_chart_race(df, n_bars=8, bar_size=.99)

    def test_period_label(self):
        bar_chart_race(df,  n_bars=8, period_label=False)
        bar_chart_race(df, n_bars=8, period_label={'x': .99, 'y': .1, 'ha': 'right'})

    def test_period_fmt(self):
        bar_chart_race(df, n_bars=8, period_template='%b %d, %Y')
        bar_chart_race(df1, n_bars=8, interpolate_period=True, period_template='{x: .2f}')

    def test_period_summary_func(self):
        def summary(values, ranks):
            total_deaths = int(round(values.sum(), -2))
            s = f'Total Deaths - {total_deaths:,.0f}'
            return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}

        bar_chart_race(df, n_bars=8, period_summary_func=summary)
    
    def test_perpendicular_bar_func(self):
        bar_chart_race(df, n_bars=8, perpendicular_bar_func='mean')
        def func(values, ranks):
            return values.quantile(.9)
        
        bar_chart_race(df, n_bars=8, perpendicular_bar_func=func)

    def test_period_length(self):
        bar_chart_race(df, n_bars=8, period_length=1200)

    def test_figsize(self):
        bar_chart_race(df, fig_kwargs={'figsize': (4, 2), 'dpi': 120})

    def test_filter_column_colors(self):
        with pytest.warns(UserWarning):
            bar_chart_race(df, n_bars=6, sort='asc', colors='Accent')

        bar_chart_race(df,  n_bars=6, sort='asc', colors='Accent', filter_column_colors=True)
        bar_chart_race(df, n_bars=6, colors=plt.cm.tab20.colors[:19])

    def test_colors(self):
        bar_chart_race(df, colors=['red', 'blue'], filter_column_colors=True)

        with pytest.raises(KeyError):
            bar_chart_race(df, colors='adf')

    def test_title(self):
        bar_chart_race(df, n_bars=6, title='Great title')
        bar_chart_race(df, n_bars=6, title={'label': 'Great title', 'size':20})

    def test_shared_fontdict(self):
        bar_chart_race(df, n_bars=6, shared_fontdict={'family': 'Courier New', 
                                            'weight': 'bold', 'color': 'teal'})

    def test_scale(self):
        bar_chart_race(df, n_bars=6, scale='log')

    def test_save(self):
        bar_chart_race(df, 'tests/videos/test.mp4', n_bars=6)
        bar_chart_race(df, 'tests/videos/test.gif', n_bars=6, writer='imagemagick')
        bar_chart_race(df, 'tests/videos/test.html', n_bars=6)

    def test_writer(self):
        bar_chart_race(df, 'tests/videos/test.mpeg', n_bars=6, writer='imagemagick')

    def test_fig(self):
        fig, ax = plt.subplots(dpi=100)
        bar_chart_race(df, n_bars=6, fig=fig)

    def test_bar_kwargs(self):
        bar_chart_race(df, n_bars=6, bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})
        