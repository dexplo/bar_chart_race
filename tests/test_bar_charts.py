import pytest
import matplotlib.pyplot as plt
import bar_chart_race as bcr


class TestSimpleBC:

    df = bcr.load_dataset('covid19')
    df = df.iloc[-20:-16]
    df1 = df.reset_index(drop=True)

    def test_defaults(self):
        bcr.bar_chart_race(self.df)
        bcr.bar_chart_race(self.df, orientation='v')

    def test_sort(self):
        bcr.bar_chart_race(self.df, sort='asc')
        bcr.bar_chart_race(self.df, orientation='v', sort='asc')

    def test_nbars(self):
        bcr.bar_chart_race(self.df, sort='desc', n_bars=8)
        bcr.bar_chart_race(self.df, orientation='v', sort='desc', n_bars=8)

    def test_fixed_order(self):
        bcr.bar_chart_race(self.df), sort='asc', n_bars=8, fixed_order=True)
        bcr.bar_chart_race(self.df, fixed_order=['Iran', 'USA', 'Italy', 'Spain'])

    def test_fixed_max(self):
        bcr.bar_chart_race(self.df, fixed_max=True)

    def test_steps_per_period(self):
        bcr.bar_chart_race(self.df, sort='asc', steps_per_period=2)
        bcr.bar_chart_race(self.df, sort='asc', steps_per_period=30)

    def test_interpolate_period(self):
        bcr.bar_chart_race(self.df, interpolate_period=True, figsize=(5, 3), n_bars=8)

    def test_label_bars(self):
        bcr.bar_chart_race(self.df, figsize=(5, 3), n_bars=8, label_bars=False)

    def test_bar_size(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, bar_size=.99)

    def test_period_label(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_label=False)
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, 
                           period_label={'x': .99, 'y': .1, 'ha': 'right'})

    def test_period_fmt(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_fmt='%b %-d, %Y')
        bcr.bar_chart_race(self.df1, figsize=(4, 2.5), n_bars=8, interpolate_period=True, 
                           period_fmt='{x: .2f}')

    def test_period_summary_func(self):
        def summary(values, ranks):
            total_deaths = int(round(values.sum(), -2))
            s = f'Total Deaths - {total_deaths:,.0f}'
            return {'x': .99, 'y': .05, 's': s, 'ha': 'right', 'size': 8}

        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_summary_func=summary)
    
    def test_perpendicular_bar_func(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_summary_func=summary,
                      perpendicular_bar_func='mean')
        def func(values, ranks):
            return values.quantile(.9)
        
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_summary_func=summary,
                            perpendicular_bar_func=func)

    def test_period_length(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=8, period_length=1200)

    def test_figsize(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5))

    def test_filter_column_colors(self):
        with pytest.warns(UserWarning):
            bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, sort='asc', cmap='Accent')

        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, sort='asc', cmap='Accent', 
                          filter_column_colors=True)

        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, cmap=plt.cm.tab20.colors[:19])

    def test_cmap(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), cmap=['red', 'blue'], 
                           filter_column_colors=True)

        with pytest.raises(KeyError):
            bcr.bar_chart_race(self.df, cmap='adf')

    def test_title(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, title='Great title', title_size=4)
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, title='Great title', 
                           title_size='xx-large')
    
    def test_label_size(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, 
                           bar_label_size=4, tick_label_size=12)

    def test_shared_fontdict(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, 
                   shared_fontdict={'family': 'Courier New', 'weight': 'bold', 'color': 'teal'}))

    def test_scale(self):
        bcr.bar_chart_race(self.df, figsize=(4, 2.5), n_bars=6, scale='log')

    def test_save(self):
        bcr.bar_chart_race(self.df, 'videos/test.mp4', figsize=(4, 2.5), n_bars=6)
        bcr.bar_chart_race(self.df, 'videos/test.gif', figsize=(4, 2.5), n_bars=6)
        bcr.bar_chart_race(self.df, 'videos/test.html', figsize=(4, 2.5), n_bars=6)

    def test_writer(self):
        bcr.bar_chart_race(self.df, 'videos/test.mpeg', figsize=(4, 2.5), n_bars=6, 
                           writer='imagemagick')

    def test_fig(self):
        fig, ax = plt.subplots(dpi=100)
        bcr.bar_chart_race(self.df, n_bars=6, fig=fig)

    def test_dpi(self):
        bcr.bar_chart_race(self.df, n_bars=6, dpi=90)

    def test_bar_kwargs(self):
        bcr.bar_chart_race(self.df, n_bars=6, bar_kwargs={'alpha': .2, 'ec': 'black', 'lw': 3})