import pytest
from bar_chart_race import load_dataset, bar_chart_race_plotly


df = load_dataset('covid19')
df = df.iloc[-20:-16]
df1 = df.reset_index(drop=True)



class TestSimpleBC:

    
    def test_defaults(self):
        bar_chart_race_plotly(df)
        bar_chart_race_plotly(df, orientation='v')

    def test_sort(self):
        bar_chart_race_plotly(df, sort='asc')
        bar_chart_race_plotly(df, orientation='v', sort='asc')

    def test_nbars(self):
        bar_chart_race_plotly(df, sort='desc', n_bars=8)
        bar_chart_race_plotly(df, orientation='v', sort='desc', n_bars=8)

    def test_fixed_order(self):
        bar_chart_race_plotly(df, sort='asc', n_bars=8, fixed_order=True)
        bar_chart_race_plotly(df, fixed_order=['Iran', 'USA', 'Italy', 'Spain'])

    def test_fixed_max(self):
        bar_chart_race_plotly(df, fixed_max=True)

    def test_steps_per_period(self):
        bar_chart_race_plotly(df, sort='asc', steps_per_period=2)
        bar_chart_race_plotly(df, sort='asc', steps_per_period=30)

    def test_interpolate_period(self):
        bar_chart_race_plotly(df, interpolate_period=True, n_bars=8)

    def test_textposition(self):
        bar_chart_race_plotly(df, n_bars=8, textposition='inside')

    def test_bar_size(self):
        bar_chart_race_plotly(df, n_bars=8, bar_size=.99)

    def test_period_label(self):
        bar_chart_race_plotly(df, n_bars=8, period_label=False)
        bar_chart_race_plotly(df, n_bars=8, 
                           period_label={'x': .99, 'y': .1, 'ha': 'right'})

    def test_period_fmt(self):
        bar_chart_race_plotly(df, n_bars=8, period_fmt='%b %-d, %Y')
        bar_chart_race_plotly(df1, n_bars=8, interpolate_period=True, 
                           period_fmt='{x: .2f}')

    def test_period_summary_func(self):
        def summary(values, ranks):
            total_deaths = int(round(values.sum(), -2))
            s = f'Total Deaths - {total_deaths:,.0f}'
            return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}

        bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary)
    
    def test_perpendicular_bar_func(self):
        bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary,
                      perpendicular_bar_func='mean')
        def func(values, ranks):
            return values.quantile(.9)
        
        bar_chart_race_plotly(df, n_bars=8, period_summary_func=summary,
                            perpendicular_bar_func=func)

    def test_period_length(self):
        bar_chart_race_plotly(df, n_bars=8, period_length=1200)

    def test_figsize(self):
        bar_chart_race_plotly(df, figsize=(4, 2.5))

    def test_filter_column_colors(self):
        with pytest.warns(UserWarning):
            bar_chart_race_plotly(df, n_bars=6, sort='asc', cmap='Accent')

        bar_chart_race_plotly(df, n_bars=6, sort='asc', cmap='Accent', 
                          filter_column_colors=True)

        bar_chart_race_plotly(df, n_bars=6, cmap=plt.cm.tab20.colors[:19])

    def test_cmap(self):
        bar_chart_race_plotly(df, cmap=['red', 'blue'], 
                           filter_column_colors=True)

        with pytest.raises(KeyError):
            bar_chart_race_plotly(df, cmap='adf')

    def test_title(self):
        bar_chart_race_plotly(df, n_bars=6, title='Great title', title_size=4)
        bar_chart_race_plotly(df, n_bars=6, title='Great title', 
                           title_size='xx-large')
    
    def test_label_size(self):
        bar_chart_race_plotly(df, n_bars=6, 
                           bar_label_size=4, tick_label_size=12)

    def test_shared_fontdict(self):
        bar_chart_race_plotly(df, n_bars=6, 
                   shared_fontdict={'family': 'Courier New', 'weight': 'bold', 'color': 'teal'}))

    def test_scale(self):
        bar_chart_race_plotly(df, n_bars=6, scale='log')

    def test_save(self):
        bar_chart_race_plotly(df, 'videos/test.mp4', n_bars=6)
        bar_chart_race_plotly(df, 'videos/test.gif', n_bars=6)
        bar_chart_race_plotly(df, 'videos/test.html', n_bars=6)

    def test_writer(self):
        bar_chart_race_plotly(df, 'videos/test.mpeg', n_bars=6, 
                           writer='imagemagick')

    def test_fig(self):
        fig, ax = plt.subplots(dpi=100)
        bar_chart_race_plotly(df, n_bars=6, fig=fig)

    def test_dpi(self):
        bar_chart_race_plotly(df, n_bars=6, dpi=90)

    def test_bar_kwargs(self):
        bar_chart_race_plotly(df, n_bars=6, bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})